from typing import TYPE_CHECKING, Callable, Optional, cast

import cv2
import numpy as np
from pyqttoast import Toast, ToastPreset, ToastPosition
from PySide6 import QtWidgets, QtCore, QtGui

from app.ui.widgets import widget_components
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
import app.helpers.miscellaneous as misc_helpers
from app.helpers.miscellaneous import get_video_rotation
from app.helpers.typing_helper import ControlTypes

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


@QtCore.Slot(str, str, QtWidgets.QWidget)
def create_and_show_messagebox(
    main_window: "MainWindow",
    window_title: str,
    message: str,
    parent_widget: QtWidgets.QWidget,
):
    messagebox = QtWidgets.QMessageBox(parent_widget)
    messagebox.setWindowTitle(window_title)
    messagebox.setWindowIcon(QtGui.QIcon(":/media/media/visomaster_small.png"))

    messagebox.setText(message)
    messagebox.exec_()


def create_and_show_toast_message(
    main_window: "MainWindow", title: str, message: str, style_type="information"
):
    style_preset_map = {
        "success": ToastPreset.SUCCESS,
        "warning": ToastPreset.WARNING,
        "error": ToastPreset.ERROR,
        "information": ToastPreset.INFORMATION,
        "success_dark": ToastPreset.SUCCESS_DARK,
        "warning_dark": ToastPreset.WARNING_DARK,
        "error_dark": ToastPreset.ERROR_DARK,
        "information_dark": ToastPreset.INFORMATION_DARK,
    }
    toast = Toast(main_window)
    toast.setTitle(title)
    toast.setText(message)
    toast.setDuration(1400)
    toast.setPosition(ToastPosition.TOP_RIGHT)  # Default: ToastPosition.BOTTOM_RIGHT
    toast.applyPreset(style_preset_map[style_type])  # Apply style preset
    toast.show()


def create_control(main_window: "MainWindow", control_name, control_value):
    main_window.control[control_name] = control_value


def update_control(
    main_window: "MainWindow",
    control_name,
    control_value,
    exec_function: Optional[Callable] = None,
    exec_function_args: Optional[list] = None,
):
    exec_function_args = exec_function_args or []
    current_position = main_window.videoSeekSlider.value()

    # Update marker control too
    # Do not update values of control with exec_function (like max threads count) as it would slow down the app heavily
    if main_window.markers.get(current_position) and not exec_function:
        main_window.markers[current_position]["control"][control_name] = control_value

    if exec_function:
        # Only execute the function if the value is different from current
        if main_window.control[control_name] != control_value:
            # By default an exec function definition should have atleast one parameter : MainWindow
            exec_function_args = [main_window, control_value] + exec_function_args
            exec_function(*exec_function_args)
    main_window.control[control_name] = control_value
    # Also update the feeder's state if it's running
    if (
        main_window.video_processor.processing
        and main_window.video_processor.feeder_control
    ):
        with main_window.video_processor.state_lock:
            # Cast to ControlTypes to satisfy the type checker, as feeder_control is typed
            if control_name in cast(
                ControlTypes, main_window.video_processor.feeder_control
            ):
                cast(ControlTypes, main_window.video_processor.feeder_control)[
                    control_name
                ] = control_value
    refresh_frame(main_window)


def create_default_parameter(
    main_window: "MainWindow", parameter_name, parameter_value
):
    main_window.default_parameters[parameter_name] = parameter_value


def create_parameter_dict_for_face_id(main_window: "MainWindow", face_id: str):
    if not main_window.parameters.get(face_id):
        parameters = (
            main_window.parameters.get(main_window.selected_target_face_id)
            or main_window.current_widget_parameters
            or main_window.default_parameters
        )
        if isinstance(parameters, dict):
            parameters = misc_helpers.ParametersDict(
                parameters, main_window.default_parameters
            )
        main_window.parameters[face_id] = parameters.copy()
    # print("Created parameter_dict_for_face_id", face_id)


def update_parameter(
    main_window: "MainWindow",
    parameter_name,
    parameter_value,
    enable_refresh_frame=True,
    exec_function: Optional[Callable] = None,
    exec_function_args: Optional[list] = None,
):
    exec_function_args = exec_function_args or []
    current_position = main_window.videoSeekSlider.value()
    face_id = main_window.selected_target_face_id

    # Get the old value for comparison before any updates
    old_parameter_value = None
    if (
        main_window.target_faces
        and face_id
        and parameter_name in main_window.parameters[face_id]
    ):
        old_parameter_value = main_window.parameters[face_id][parameter_name]
    elif (
        main_window.current_widget_parameters
        and parameter_name in main_window.current_widget_parameters
    ):
        old_parameter_value = main_window.current_widget_parameters[parameter_name]

    # --- Update the data dictionaries ---

    # Update marker parameters if applicable
    if main_window.markers.get(current_position) and face_id:
        main_window.markers[current_position]["parameters"][face_id][parameter_name] = (
            parameter_value
        )

    # Update parameters for the selected face
    if main_window.target_faces and face_id:
        main_window.parameters[face_id][parameter_name] = parameter_value
        # Also update the feeder's state if it's running
        if (
            main_window.video_processor.processing
            and main_window.video_processor.feeder_parameters
        ):
            with main_window.video_processor.state_lock:
                if face_id in main_window.video_processor.feeder_parameters:
                    main_window.video_processor.feeder_parameters[face_id][
                        parameter_name
                    ] = parameter_value

    # Always update the current widget state
    if main_window.current_widget_parameters:
        main_window.current_widget_parameters[parameter_name] = parameter_value

    # --- Trigger actions ---

    # Refresh the frame if needed
    if enable_refresh_frame:
        refresh_frame(main_window)

    # Execute the associated function if the value has changed
    # This now runs even if no face is selected, fixing the unload issue.
    if exec_function and parameter_value != old_parameter_value:
        # The first argument is always the main_window, followed by the new value
        final_exec_args: list = [main_window, parameter_value] + exec_function_args
        exec_function(*final_exec_args)


def refresh_frame(main_window: "MainWindow"):
    video_processor = main_window.video_processor
    if not video_processor.processing:
        video_processor.process_current_frame()


# Function to Hide Elements conditionally from values in LayoutData (Currently supports using Selection box and Toggle button to hide other widgets)
def show_hide_related_widgets(
    main_window: "MainWindow",
    parent_widget,
    parent_widget_name: str,
    value1=False,
    value2=False,
):
    if main_window.parameter_widgets:
        group_layout_data = parent_widget.group_layout_data  # Dictionary contaning layout data of all elements in the group of the parent_widget

        # --- CASE 1: PARENT IS A SELECTION BOX (e.g., Simple/Advanced) ---
        if "Selection" in parent_widget_name:
            for widget_name in group_layout_data.keys():
                current_widget = main_window.parameter_widgets.get(widget_name)
                layout_info = group_layout_data[widget_name]

                # Only process widgets that depend on THIS selection box
                if (
                    layout_info.get("parentSelection", "") == parent_widget_name
                    and current_widget
                ):
                    # 1. Check Selection Condition
                    selection_condition_met = (
                        layout_info.get("requiredSelectionValue")
                        == parent_widget.currentText()
                    )

                    # 2. Check Toggle Condition (Cross-Check)
                    # Even if selection matches, we must check if the parent toggles are ON
                    toggle_condition_met = True
                    parentToggles = layout_info.get("parentToggle", "")

                    if parentToggles and selection_condition_met:
                        if "&" in parentToggles:
                            toggles = [t.strip() for t in parentToggles.split("&")]
                            for t_name in toggles:
                                t_widget = main_window.parameter_widgets.get(t_name)
                                if t_widget and not t_widget.isChecked():
                                    toggle_condition_met = False
                                    break
                        elif "|" in parentToggles:
                            toggle_condition_met = False
                            toggles = [t.strip() for t in parentToggles.split("|")]
                            for t_name in toggles:
                                t_widget = main_window.parameter_widgets.get(t_name)
                                if t_widget and t_widget.isChecked():
                                    toggle_condition_met = True
                                    break
                        else:
                            # Single toggle or simple logic
                            t_widget = main_window.parameter_widgets.get(parentToggles)
                            required_val = layout_info.get("requiredToggleValue", True)
                            if t_widget and t_widget.isChecked() != required_val:
                                toggle_condition_met = False

                    # Final Decision
                    if selection_condition_met and toggle_condition_met:
                        current_widget.show()
                        current_widget.label_widget.show()
                        current_widget.reset_default_button.show()
                        if current_widget.line_edit:
                            current_widget.line_edit.show()
                    else:
                        current_widget.hide()
                        current_widget.label_widget.hide()
                        current_widget.reset_default_button.hide()
                        if current_widget.line_edit:
                            current_widget.line_edit.hide()

        # --- CASE 2: PARENT IS A TOGGLE BUTTON ---
        elif "Toggle" in parent_widget_name:
            for widget_name in group_layout_data.keys():
                if widget_name not in main_window.parameter_widgets:
                    continue
                current_widget = main_window.parameter_widgets[widget_name]
                layout_info = group_layout_data[widget_name]

                parentToggles = layout_info.get("parentToggle", "")

                # Only process widgets that depend on THIS toggle (or have it in their chain)
                if parent_widget_name in parentToggles:
                    # 1. Check Selection Condition (Cross-Check)
                    # Before evaluating toggles, check if the parent Selection is valid
                    selection_condition_met = True
                    parentSelection = layout_info.get("parentSelection", "")
                    if parentSelection:
                        sel_widget = main_window.parameter_widgets.get(parentSelection)
                        if sel_widget and sel_widget.currentText() != layout_info.get(
                            "requiredSelectionValue"
                        ):
                            selection_condition_met = False

                    # 2. Check Toggle Condition
                    toggle_condition_met = False

                    if "," in parentToggles:
                        # Legacy comma logic (iterative check)
                        result = [item.strip() for item in parentToggles.split(",")]
                        parentToggle_ischecked = False
                        for _, required_widget_name in enumerate(result):
                            w = main_window.parameter_widgets.get(required_widget_name)
                            if w:
                                parentToggle_ischecked = w.isChecked()

                        if (
                            layout_info.get("requiredToggleValue")
                            == parentToggle_ischecked
                        ):
                            toggle_condition_met = True

                    elif "|" in parentToggles:
                        # OR Logic
                        result = [item.strip() for item in parentToggles.split("|")]
                        any_checked = False
                        for required_widget_name in result:
                            w = main_window.parameter_widgets.get(required_widget_name)
                            if w and w.isChecked():
                                any_checked = True
                                break

                        if layout_info.get("requiredToggleValue") == any_checked:
                            toggle_condition_met = True

                    elif "&" in parentToggles:
                        # AND Logic
                        result = [item.strip() for item in parentToggles.split("&")]
                        all_checked = True
                        for required_widget_name in result:
                            w = main_window.parameter_widgets.get(required_widget_name)
                            if w and not w.isChecked():
                                all_checked = False
                                break

                        if layout_info.get("requiredToggleValue") == all_checked:
                            toggle_condition_met = True

                    else:
                        # Single Toggle
                        w = main_window.parameter_widgets.get(parentToggles)
                        parentToggle_ischecked = w.isChecked() if w else False
                        if (
                            layout_info.get("requiredToggleValue")
                            == parentToggle_ischecked
                        ):
                            toggle_condition_met = True

                    # Final Decision
                    if selection_condition_met and toggle_condition_met:
                        current_widget.show()
                        current_widget.label_widget.show()
                        current_widget.reset_default_button.show()
                        if current_widget.line_edit:
                            current_widget.line_edit.show()
                    else:
                        current_widget.hide()
                        current_widget.label_widget.hide()
                        current_widget.reset_default_button.hide()
                        if current_widget.line_edit:
                            current_widget.line_edit.hide()

            parent_widget.start_animation()


# @misc_helpers.benchmark
def get_pixmap_from_frame(main_window: "MainWindow", frame: np.ndarray):
    height, width, channel = frame.shape
    if channel == 2:
        # Frame in grayscale
        bytes_per_line = width
        q_img = QtGui.QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_Grayscale8,
        )
    else:
        # Frame in color
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(
            frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888
        ).rgbSwapped()
    pixmap = QtGui.QPixmap.fromImage(q_img)
    return pixmap


def update_gpu_memory_progressbar(main_window: "MainWindow"):
    _update_gpu_memory_progressbar(main_window)


def _update_gpu_memory_progressbar(main_window: "MainWindow"):
    memory_used, memory_total = main_window.models_processor.get_gpu_memory()
    main_window.gpu_memory_update_signal.emit(memory_used, memory_total)


@QtCore.Slot(int, int)
def set_gpu_memory_progressbar_value(
    main_window: "MainWindow", memory_used, memory_total
):
    main_window.vramProgressBar.setMaximum(memory_total)
    main_window.vramProgressBar.setValue(memory_used)
    main_window.vramProgressBar.setFormat(
        f"{round(memory_used / 1024, 2)} GB / {round(memory_total / 1024, 2)} GB (%p%)"
    )
    # Base style copied from true_dark_styles.qss
    base_style = """
        QProgressBar {
            border: 1px solid #363636;
            border-radius: 5px;
            text-align: center;
            background-color: #202020;
            color: #f0f0f0;
        }
    """

    # Base Chunk style copied from true_dark_styles.qss
    chunk_style_normal = """
        QProgressBar::chunk {
            background-color: #16759e;  /* Blue */
            width: 10px;
            margin: 0.5px;
            border-radius: 4px;
        }
    """

    # High VRAM Chunck style
    chunk_style_high = """
        QProgressBar::chunk {
            background-color: #911414; /* Red */
            width: 10px;
            margin: 0.5px;
            border-radius: 4px;
        }
    """

    if (memory_used / memory_total) > 0.85:
        main_window.vramProgressBar.setStyleSheet(base_style + chunk_style_high)
    else:
        main_window.vramProgressBar.setStyleSheet(base_style + chunk_style_normal)

    main_window.vramProgressBar.update()


def clear_gpu_memory(main_window: "MainWindow"):
    main_window.video_processor.stop_processing()
    main_window.models_processor.clear_gpu_memory()
    main_window.swapfacesButton.setChecked(False)
    main_window.editFacesButton.setChecked(False)
    update_gpu_memory_progressbar(main_window)

    # main_window.videoSeekSlider.markers = set() # Comment this to keep markers visible after vram clear
    main_window.videoSeekSlider.update()


def extract_frame_as_pixmap(
    main_window: "MainWindow",
    media_file_path,
    file_type,
    webcam_index=False,
    webcam_backend=False,
):
    """
    Extracts a frame from a media file and converts it to a QPixmap for thumbnails.
    It now uses the ThumbnailManager to efficiently cache and retrieve thumbnails.
    """

    # This helper function converts a numpy frame to a scaled QPixmap.
    def convert_frame_to_pixmap(frame):
        height, width, _ = frame.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(
            frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        # The final scaling for display is consistent.
        return pixmap.scaled(70, 70, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    # For images and videos, first check for a cached thumbnail.
    if file_type in ["image", "video"]:
        # We use the thumbnail_manager instance from the main_window.
        thumbnail_path = main_window.thumbnail_manager.find_existing_thumbnail(
            media_file_path
        )

        if thumbnail_path:
            frame = misc_helpers.read_image_file(thumbnail_path)
            if frame is not None:
                return convert_frame_to_pixmap(frame)

    # If no cache is found, or for webcams, generate the frame from source.
    frame = None
    if file_type == "image":
        frame = misc_helpers.read_image_file(media_file_path)
    elif file_type == "video":
        # MODIFICATION: Get rotation for thumbnail
        rotation_angle = get_video_rotation(media_file_path)
        cap = cv2.VideoCapture(media_file_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame_no = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_no)
            ret, frame = misc_helpers.read_frame(cap, rotation_angle)
            cap.release()
    elif file_type == "webcam":
        camera = cv2.VideoCapture(webcam_index, webcam_backend)
        if camera.isOpened():
            # MODIFICATION: Pass 0 for webcam rotation
            ret, frame = misc_helpers.read_frame(camera, 0)
            camera.release()  # Release camera immediately after grabbing one frame

    if isinstance(frame, np.ndarray):
        # Create a new thumbnail in the cache for next time.
        if file_type != "webcam":
            main_window.thumbnail_manager.create_thumbnail(frame, media_file_path)

        # Return the generated pixmap.
        return convert_frame_to_pixmap(frame)

    return None  # Return None if everything failed.


def set_widgets_values_using_face_id_parameters(
    main_window: "MainWindow", face_id=False
):
    if (face_id is False) or (not main_window.parameters.get(face_id)):
        # print("Set widgets values using default parameters")
        if main_window.current_widget_parameters:
            parameters = main_window.current_widget_parameters.copy()
        else:
            parameters = main_window.default_parameters
    else:
        # print(f"Set widgets values using face_id {face_id}")
        parameters = main_window.parameters[face_id].copy()
    parameter_widgets = main_window.parameter_widgets
    for parameter_name, parameter_value in parameters.items():
        widget = parameter_widgets.get(parameter_name)
        if widget:
            # temporarily disable refreshing the frame to prevent slowing due to unnecessary processing
            widget.enable_refresh_frame = False
            if isinstance(
                widget,
                (
                    widget_components.ParameterLineEdit,
                    widget_components.ParameterSlider,
                ),
            ):
                try:
                    int_value = int(float(parameter_value))
                    widget.set_value(int_value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(
                widget,
                (
                    widget_components.ParameterLineDecimalEdit,
                    widget_components.ParameterDecimalSlider,
                ),
            ):
                try:
                    float_value = float(parameter_value)
                    widget.set_value(float_value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(widget, widget_components.ToggleButton):
                widget.set_value(bool(parameter_value))
            elif isinstance(widget, widget_components.SelectionBox):
                widget.set_value(str(parameter_value))
            else:
                widget.set_value(parameter_value)
            widget.enable_refresh_frame = True


def set_control_widgets_values(main_window: "MainWindow", enable_exec_func=True):
    """
    Set the values of control widgets based on the `control` data in the `main_window`.

    Temporarily disables frame refreshing while setting values to avoid unnecessary processing.
    """
    # Get control values and parameter widgets from the main window
    control = main_window.control.copy()
    parameter_widgets = main_window.parameter_widgets

    # Prepare a dictionary of ALL widget options from layout data
    all_widget_options = {}
    for layout_data_source in [
        SETTINGS_LAYOUT_DATA,
        COMMON_LAYOUT_DATA,
    ]:  # Iterate over both
        for group_name, group_data in layout_data_source.items():
            for widget_key, widget_data in group_data.items():
                all_widget_options[widget_key] = widget_data

    # Iterate through control items and update widgets
    for control_name, control_value in control.items():
        widget = parameter_widgets.get(control_name)

        if widget:
            # Temporarily disable frame refresh
            widget.enable_refresh_frame = False

            # Set the widget value
            if isinstance(
                widget,
                (
                    widget_components.ParameterLineEdit,
                    widget_components.ParameterSlider,
                ),
            ):
                try:
                    int_value = int(float(control_value))
                    widget.set_value(int_value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(
                widget,
                (
                    widget_components.ParameterLineDecimalEdit,
                    widget_components.ParameterDecimalSlider,
                ),
            ):
                try:
                    float_value = float(control_value)
                    widget.set_value(float_value)
                except (ValueError, TypeError):
                    pass
            elif isinstance(widget, widget_components.ToggleButton):
                widget.set_value(bool(control_value))
            elif isinstance(widget, widget_components.SelectionBox):
                widget.set_value(str(control_value))
            else:
                widget.set_value(control_value)

            if enable_exec_func:
                # Execute any associated function, if defined
                widget_definition = all_widget_options.get(
                    control_name
                )  # Use .get() for safety
                if widget_definition:
                    exec_function_data = widget_definition.get("exec_function")
                    if exec_function_data:
                        # The functions in control_actions.py are typically (main_window, value, *additional_args)
                        exec_args_from_layout = widget_definition.get(
                            "exec_function_args", []
                        )
                        final_exec_args = [
                            main_window,
                            control_value,
                        ] + exec_args_from_layout
                        exec_function_data(*final_exec_args)

            # Re-enable frame refresh
            widget.enable_refresh_frame = True


@QtCore.Slot(QtWidgets.QListWidget, bool)
def update_placeholder_visibility(
    main_window: "MainWindow", list_widget: QtWidgets.QListWidget, default_hide
):
    # """Update the visibility of the placeholder text."""
    # """
    #     The default_hide parameter is used to Hide the placeholder text by default.
    #     If the default_hide is False, then the visibility of the placeholder text is set using the size of the list_widget
    # """
    if default_hide:
        is_visible = False
    else:
        is_visible = list_widget.count() == 0
    list_widget.placeholder_label.setVisible(is_visible)
    # Set Cursor on the List Widget
    if is_visible:
        list_widget.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    else:
        list_widget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
    # print("SetVisible", is_visible)
    # print("targetVideosList.count()", list_widget.count())


@QtCore.Slot()
def show_model_loading_dialog(main_window: "MainWindow"):
    # Debounce: Only show dialog if loading takes longer than 300ms
    if not hasattr(main_window, "_model_loading_timer"):
        main_window._model_loading_timer = QtCore.QTimer()
        main_window._model_loading_timer.setSingleShot(True)

        def show_dialog():
            if (
                not hasattr(main_window, "model_loading_dialog")
                or main_window.model_loading_dialog is None
            ):
                main_window.model_loading_dialog = widget_components.LoadingDialog()
            if not main_window.model_loading_dialog.isVisible():
                main_window.model_loading_dialog.show()
                QtWidgets.QApplication.processEvents()

        main_window._model_loading_timer.timeout.connect(show_dialog)
    # Start or restart the timer
    main_window._model_loading_timer.start(300)


@QtCore.Slot()
def hide_model_loading_dialog(main_window: "MainWindow"):
    # Stop the timer if it's running
    if hasattr(main_window, "_model_loading_timer"):
        main_window._model_loading_timer.stop()
    # Only hide if dialog exists and is visible
    if (
        hasattr(main_window, "model_loading_dialog")
        and main_window.model_loading_dialog is not None
    ):
        if main_window.model_loading_dialog.isVisible():
            main_window.model_loading_dialog.hide()
            QtWidgets.QApplication.processEvents()
