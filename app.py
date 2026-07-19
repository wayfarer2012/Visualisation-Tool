"""A simple desktop app for saving images by phone number."""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QPointF, QProcess, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HoverSegmentItem(QGraphicsItem):
    """One selectable, hoverable, and colourable room-surface overlay."""

    # Rendering parameters can be tuned by architectural surface type without
    # changing saved colour data or any editor workflow.
    RECOLOUR_SETTINGS = {
        "wall": {
            "colour_blend_strength": 0.94,
            "black_layer_strength": 0.88,
            "white_layer_strength": 0.72,
            "detail_layer_strength": 0.72,
            "contrast_layer_strength": 0.18,
            "saturation_adjustment": 0.92,
            "edge_feather_radius": 2,
        },
        "pillar": {
            "colour_blend_strength": 0.95,
            "black_layer_strength": 0.92,
            "white_layer_strength": 0.78,
            "detail_layer_strength": 0.78,
            "contrast_layer_strength": 0.20,
            "saturation_adjustment": 0.94,
            "edge_feather_radius": 1,
        },
        "trim": {
            "colour_blend_strength": 0.96,
            "black_layer_strength": 0.94,
            "white_layer_strength": 0.84,
            "detail_layer_strength": 0.86,
            "contrast_layer_strength": 0.24,
            "saturation_adjustment": 0.96,
            "edge_feather_radius": 1,
        },
        "ceiling": {
            "colour_blend_strength": 0.90,
            "black_layer_strength": 0.82,
            "white_layer_strength": 0.76,
            "detail_layer_strength": 0.62,
            "contrast_layer_strength": 0.14,
            "saturation_adjustment": 0.88,
            "edge_feather_radius": 2,
        },
    }

    def __init__(
        self,
        segment_id: str,
        segment_type: str,
        shape_type: str,
        shape_path: QPainterPath,
        selection_changed_callback,
        original_image: QImage,
        points: list | None = None,
        mask_path: str | None = None,
        mask_image: QImage | None = None,
        applied_colour: str | None = None,
    ) -> None:
        super().__init__()

        self.segment_id = segment_id
        self.segment_type = segment_type
        self.shape_type = shape_type
        self.shape_path = shape_path
        self.points = points
        self.mask_path = mask_path
        self.mask_image = mask_image
        self.is_selected = False
        self.applied_colour = applied_colour
        self.selection_changed_callback = selection_changed_callback
        self.original_image = original_image
        self.is_hovered = False
        self.tint_image = None
        self.tint_image_position = QPointF()

        self.setAcceptHoverEvents(True)
        self.setZValue(1)  # Keep the segment above the image.
        self.rebuild_realistic_tint()
        self.refresh_appearance()

    def boundingRect(self) -> QRectF:
        """Return the smallest scene rectangle containing this segment."""
        return self.shape_path.boundingRect()

    def shape(self) -> QPainterPath:
        """Use polygon points or mask pixels as the selectable hover region."""
        return self.shape_path

    def hoverEnterEvent(self, event) -> None:
        """Show a white highlight while preserving any realistic colour tint."""
        self.is_hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Restore the segment's normal, selected, or coloured appearance."""
        self.is_hovered = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        """Toggle this segment while allowing other segments to stay selected."""
        self.is_selected = not self.is_selected
        self.refresh_appearance()
        self.selection_changed_callback()
        super().mousePressEvent(event)

    def set_applied_colour(self, colour: str | None) -> None:
        """Apply or remove a hex colour and refresh the visible overlay."""
        self.applied_colour = colour
        self.rebuild_realistic_tint()
        self.refresh_appearance()

    def refresh_appearance(self) -> None:
        """Refresh the selection outline and trigger custom tint painting."""
        self.update()

    @staticmethod
    def srgb_to_linear(channel: int) -> float:
        """Convert one 8-bit sRGB channel into linear-light RGB."""
        value = channel / 255.0
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    @staticmethod
    def linear_to_srgb(channel: float) -> int:
        """Convert one linear-light RGB channel back to display sRGB."""
        value = max(0.0, min(1.0, channel))
        encoded = (
            value * 12.92
            if value <= 0.0031308
            else 1.055 * (value ** (1.0 / 2.4)) - 0.055
        )
        return round(encoded * 255)

    @staticmethod
    def smoothstep(edge_start: float, edge_end: float, value: float) -> float:
        """Return a smooth transition used to protect bright highlights."""
        position = max(
            0.0, min(1.0, (value - edge_start) / (edge_end - edge_start))
        )
        return position * position * (3.0 - 2.0 * position)

    @staticmethod
    def blend_normal(base: float, layer: float, opacity: float) -> float:
        """Normal alpha blend one layer channel over a base channel."""
        return base * (1.0 - opacity) + layer * opacity

    @staticmethod
    def blend_multiply(base: float, layer: float, opacity: float) -> float:
        """Multiply blend preserves dark recesses and ambient shadows."""
        multiplied = base * layer
        return HoverSegmentItem.blend_normal(base, multiplied, opacity)

    @staticmethod
    def blend_screen(base: float, layer: float, opacity: float) -> float:
        """Screen blend restores reflected light and highlight rolloff."""
        screened = 1.0 - (1.0 - base) * (1.0 - layer)
        return HoverSegmentItem.blend_normal(base, screened, opacity)

    @staticmethod
    def blend_soft_light(base: float, layer: float, opacity: float) -> float:
        """Soft Light blend restores texture without replacing paint chroma."""
        if layer <= 0.5:
            blended = base - (1.0 - 2.0 * layer) * base * (1.0 - base)
        else:
            curve = (
                ((16.0 * base - 12.0) * base + 4.0) * base
                if base <= 0.25
                else base**0.5
            )
            blended = base + (2.0 * layer - 1.0) * (curve - base)
        return HoverSegmentItem.blend_normal(base, blended, opacity)

    def build_local_mask(self, bounds, size) -> QImage:
        """Build the segment mask crop used for tinting and edge feathering."""
        if self.shape_type == "mask" and self.mask_image is not None:
            return self.mask_image.copy(bounds).convertToFormat(
                QImage.Format.Format_Grayscale8
            )

        mask = QImage(size, QImage.Format.Format_Grayscale8)
        mask.fill(0)
        painter = QPainter(mask)
        painter.translate(-bounds.left(), -bounds.top())
        painter.fillPath(self.shape_path, Qt.GlobalColor.white)
        painter.end()
        return mask

    @staticmethod
    def smoothly_scaled(image: QImage, width: int, height: int) -> QImage:
        """Resize an image with smooth interpolation for deterministic filtering."""
        return image.scaled(
            max(1, width),
            max(1, height),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    @staticmethod
    def feather_mask_edges(mask: QImage, radius: int) -> QImage:
        """Softly average mask boundaries while leaving broad interiors opaque."""
        feathered = QImage(mask.size(), QImage.Format.Format_Grayscale8)
        feathered.fill(0)
        for y in range(mask.height()):
            for x in range(mask.width()):
                total = 0
                samples = 0
                for sample_y in range(
                    max(0, y - radius), min(mask.height(), y + radius + 1)
                ):
                    for sample_x in range(
                        max(0, x - radius), min(mask.width(), x + radius + 1)
                    ):
                        total += mask.pixelColor(sample_x, sample_y).red()
                        samples += 1
                value = round(total / samples)
                feathered.setPixelColor(x, y, QColor(value, value, value))
        return feathered

    def rebuild_realistic_tint(self) -> None:
        """Build deterministic painted pixels while preserving photographic light."""
        self.tint_image = None
        if self.applied_colour is None:
            return

        settings = self.RECOLOUR_SETTINGS.get(
            self.segment_type, self.RECOLOUR_SETTINGS["wall"]
        )
        feather_radius = settings["edge_feather_radius"]
        bounds = self.shape_path.boundingRect().toAlignedRect()
        bounds.adjust(-feather_radius, -feather_radius, feather_radius, feather_radius)
        bounds = bounds.intersected(self.original_image.rect())
        if bounds.isEmpty():
            return

        source = self.original_image.copy(bounds).convertToFormat(
            QImage.Format.Format_ARGB32
        )
        local_mask = self.build_local_mask(bounds, source.size())
        tint = QImage(source.size(), QImage.Format.Format_ARGB32_Premultiplied)
        tint.fill(Qt.GlobalColor.transparent)
        chosen_colour = QColor(self.applied_colour)

        # Linear-light luminance separates illumination from the original wall
        # hue. This prevents the old paint colour from showing through.
        luminance_image = QImage(source.size(), QImage.Format.Format_Grayscale8)
        luminance_values = []
        for y in range(source.height()):
            for x in range(source.width()):
                source_colour = source.pixelColor(x, y)
                red = self.srgb_to_linear(source_colour.red())
                green = self.srgb_to_linear(source_colour.green())
                blue = self.srgb_to_linear(source_colour.blue())
                luminance = red * 0.2126 + green * 0.7152 + blue * 0.0722
                luminance_byte = round(luminance * 255)
                luminance_image.setPixelColor(
                    x, y, QColor(luminance_byte, luminance_byte, luminance_byte)
                )
                if local_mask.pixelColor(x, y).red() >= 128:
                    luminance_values.append(luminance)

        average_luminance = (
            sum(luminance_values) / len(luminance_values)
            if luminance_values
            else 0.5
        )

        # A downscale/upscale pass estimates low-frequency illumination. Dividing
        # original luminance by this shading map yields high-frequency texture.
        blur_divisor = max(8, round(min(source.width(), source.height()) / 18))
        shading_small = self.smoothly_scaled(
            luminance_image,
            max(1, source.width() // blur_divisor),
            max(1, source.height() // blur_divisor),
        )
        shading_image = self.smoothly_scaled(
            shading_small, source.width(), source.height()
        )

        # A small deterministic neighborhood filter creates an edge-aware
        # transition. Interior painted pixels stay opaque; only boundaries blend.
        feather_mask = self.feather_mask_edges(
            local_mask, settings["edge_feather_radius"]
        )

        target_red = self.srgb_to_linear(chosen_colour.red())
        target_green = self.srgb_to_linear(chosen_colour.green())
        target_blue = self.srgb_to_linear(chosen_colour.blue())
        target_luminance = max(
            0.01,
            target_red * 0.2126 + target_green * 0.7152 + target_blue * 0.0722,
        )

        # The solid paint base owns the new colour. Saturation adjustment moves
        # its channels toward neutral target luminance when a subtler paint is
        # preferred, without allowing the old wall hue to show through.
        saturation = settings["saturation_adjustment"]
        solid_paint = [
            target_luminance + (channel - target_luminance) * saturation
            for channel in (target_red, target_green, target_blue)
        ]

        for y in range(source.height()):
            for x in range(source.width()):
                mask_alpha = feather_mask.pixelColor(x, y).red() / 255.0
                if mask_alpha <= 0.01:
                    continue

                original_luminance = luminance_image.pixelColor(x, y).red() / 255.0
                shading = max(0.015, shading_image.pixelColor(x, y).red() / 255.0)

                # High-frequency detail is separated from broad illumination.
                # It later becomes a neutral Soft Light texture layer.
                detail = max(0.55, min(1.55, original_luminance / shading))
                detail_layer = max(
                    0.0, min(1.0, 0.5 + (detail - 1.0) * 0.75)
                )

                # Layer 1: solid opaque paint base. This is not the original
                # image tinted with transparency; it is a reconstructed surface.
                painted = [
                    self.blend_normal(
                        target_luminance,
                        channel,
                        settings["colour_blend_strength"],
                    )
                    for channel in solid_paint
                ]

                # Layer 2: Multiply-style black/shadow layer derived from the
                # low-frequency shading map. White is neutral; dark values add
                # corners, recesses, depth, and under-roof darkness.
                normalized_shading = max(
                    0.0, min(2.0, shading / max(average_luminance, 0.015))
                )
                shadow_amount = max(0.0, min(1.0, 1.0 - normalized_shading))
                shadow_layer = 1.0 - shadow_amount
                painted = [
                    self.blend_multiply(
                        channel,
                        shadow_layer,
                        settings["black_layer_strength"],
                    )
                    for channel in painted
                ]

                # Layer 3: Screen-style white/highlight layer restores bright
                # wall faces, reflected light, and smooth highlight rolloff.
                broad_highlight = max(
                    0.0, min(1.0, normalized_shading - 1.0)
                )
                specular_highlight = self.smoothstep(
                    0.68, 0.98, original_luminance
                )
                highlight_amount = max(broad_highlight, specular_highlight)
                painted = [
                    self.blend_screen(
                        channel,
                        highlight_amount,
                        settings["white_layer_strength"],
                    )
                    for channel in painted
                ]

                # Layer 4: neutral Soft Light detail preserves plaster, subtle
                # tonal changes, camera texture, and fine surface character.
                painted = [
                    self.blend_soft_light(
                        channel,
                        detail_layer,
                        settings["detail_layer_strength"],
                    )
                    for channel in painted
                ]

                # Layer 5: subtle grayscale Soft Light restores local contrast
                # without reintroducing the original wall colour.
                tone_layer = max(
                    0.0,
                    min(
                        1.0,
                        0.5
                        + (original_luminance - average_luminance)
                        / max(average_luminance, 0.1)
                        * 0.25,
                    ),
                )
                painted = [
                    self.blend_soft_light(
                        channel,
                        tone_layer,
                        settings["contrast_layer_strength"],
                    )
                    for channel in painted
                ]

                tinted_colour = QColor(
                    self.linear_to_srgb(painted[0]),
                    self.linear_to_srgb(painted[1]),
                    self.linear_to_srgb(painted[2]),
                    round(mask_alpha * 255),
                )
                tint.setPixelColor(x, y, tinted_colour)

        self.tint_image = tint
        self.tint_image_position = QPointF(bounds.left(), bounds.top())

    def paint(self, painter, option, widget=None) -> None:
        """Paint the clipped realistic tint, then hover and selection feedback."""
        painter.save()
        painter.setClipPath(self.shape_path)

        if self.tint_image is not None:
            painter.drawImage(self.tint_image_position, self.tint_image)

        if self.is_hovered:
            painter.fillPath(self.shape_path, QColor(255, 255, 255, 75))
        elif self.is_selected and self.applied_colour is None:
            painter.fillPath(self.shape_path, QColor(255, 255, 255, 70))

        painter.restore()

        if self.is_selected:
            painter.setPen(QPen(QColor("#ffd54f"), 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self.shape_path)

    def state(self) -> dict:
        """Return the segment state used by the simple undo/redo history."""
        return {
            "id": self.segment_id,
            "type": self.segment_type,
            "shape_type": self.shape_type,
            "selected": self.is_selected,
            "applied_colour": self.applied_colour,
        }


class ImageGraphicsView(QGraphicsView):
    """Graphics view that keeps the complete image fitted inside the window."""

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ImageUploadWindow(QMainWindow):
    """Main window containing the phone input, upload button, and image preview."""

    # -------------------------------------------------------------------------
    # UI setup
    # -------------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Phone Image Upload")
        self.resize(1100, 750)

        # The data folder is created beside this Python file.
        self.data_folder = Path(__file__).resolve().parent / "data"

        title_label = QLabel("Save an image using a phone number")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        phone_label = QLabel("Phone number:")

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Example: +1 555 123 4567")
        self.phone_input.textChanged.connect(self.reset_phone_lookup)

        self.check_phone_button = QPushButton("Check Phone Number")
        self.check_phone_button.clicked.connect(self.check_phone_number)

        phone_layout = QHBoxLayout()
        phone_layout.addWidget(self.phone_input)
        phone_layout.addWidget(self.check_phone_button)

        self.upload_button = QPushButton("Add New Image")
        self.upload_button.clicked.connect(self.select_and_save_image)
        self.upload_button.setEnabled(False)

        self.save_visualisation_button = QPushButton("Save This Visualisation")
        self.save_visualisation_button.clicked.connect(
            self.mark_current_visualisation_saved
        )
        self.save_visualisation_button.setVisible(False)

        self.view_past_button = QPushButton("View Past Visualisations")
        self.view_past_button.clicked.connect(self.show_past_visualisations)
        self.view_past_button.setVisible(False)

        self.close_past_button = QPushButton("Close Past Visualisations")
        self.close_past_button.clicked.connect(self.close_past_visualisations)
        self.close_past_button.setVisible(False)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_form)

        self.load_mock_button = QPushButton("Developer: Load Mask Mock")
        self.load_mock_button.clicked.connect(self.load_mock_mask_visualisation)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.view_past_button)
        button_layout.addWidget(self.close_past_button)
        button_layout.addWidget(self.upload_button)
        button_layout.addWidget(self.save_visualisation_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.load_mock_button)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        # The summary is refreshed from metadata after lookup, upload, or saving.
        self.phone_summary_label = QLabel("")
        self.phone_summary_label.setWordWrap(True)

        # Upload pre-processing controls operate on a temporary in-memory image.
        # Nothing is saved until the user confirms the preview.
        self.rotate_left_button = QPushButton("Rotate Left")
        self.rotate_left_button.clicked.connect(self.rotate_preview_left)

        self.rotate_right_button = QPushButton("Rotate Right")
        self.rotate_right_button.clicked.connect(self.rotate_preview_right)

        self.crop_later_button = QPushButton("Crop (Coming Later)")
        self.crop_later_button.setEnabled(False)
        self.crop_later_button.setToolTip(
            "Crop will be added later before segmentation starts."
        )

        self.confirm_image_button = QPushButton("Confirm Image")
        self.confirm_image_button.clicked.connect(self.confirm_preprocessed_image)

        self.cancel_preprocessing_button = QPushButton("Cancel")
        self.cancel_preprocessing_button.clicked.connect(self.cancel_preprocessing)

        self.preprocessing_toolbar = QWidget()
        preprocessing_layout = QHBoxLayout(self.preprocessing_toolbar)
        preprocessing_layout.setContentsMargins(0, 0, 0, 0)
        preprocessing_layout.addWidget(self.rotate_left_button)
        preprocessing_layout.addWidget(self.rotate_right_button)
        preprocessing_layout.addWidget(self.crop_later_button)
        preprocessing_layout.addWidget(self.confirm_image_button)
        preprocessing_layout.addWidget(self.cancel_preprocessing_button)
        self.preprocessing_toolbar.setVisible(False)

        # Editor controls stay hidden until a new image finishes its scan animation.
        self.hex_colour_input = QLineEdit()
        self.hex_colour_input.setPlaceholderText("Hex colour, for example #C8A27A")

        self.change_colour_button = QPushButton("Change Colour")
        self.change_colour_button.clicked.connect(self.change_selected_colour)

        self.remove_colour_button = QPushButton("Remove Colour")
        self.remove_colour_button.clicked.connect(self.remove_selected_colour)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_editor_action)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo_editor_action)

        self.close_editor_button = QPushButton("Close")
        self.close_editor_button.clicked.connect(self.close_editor_mode)

        self.editor_toolbar = QWidget()
        editor_layout = QHBoxLayout(self.editor_toolbar)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(self.hex_colour_input)
        editor_layout.addWidget(self.change_colour_button)
        editor_layout.addWidget(self.remove_colour_button)
        editor_layout.addWidget(self.undo_button)
        editor_layout.addWidget(self.redo_button)
        editor_layout.addWidget(self.close_editor_button)
        self.editor_toolbar.setVisible(False)

        # This list is used for the local past-visualisation flow.
        self.past_images_list = QListWidget()
        self.past_images_list.itemClicked.connect(self.open_saved_visualisation_from_list)
        self.past_images_list.setMinimumHeight(120)
        self.past_images_list.setVisible(False)

        # The scene holds the saved image as its base layer and demo segments above it.
        self.image_scene = QGraphicsScene()
        self.base_image_item = None
        self.current_image_path = None
        self.current_original_image = None
        self.current_segments_path = None
        self.is_mock_visualisation = False
        self.segment_items = []
        self.checked_phone_number = None
        self.visualisations = []
        self.current_visualisation_id = None
        self.editor_mode = False
        self.editor_history = []
        self.redo_history = []
        self.pending_image_path = None
        self.temporary_image_pixmap = None
        self.temporary_image_source_path = None
        self.preprocessing_mode = False
        self.scan_line_item = None
        self.scan_position = 0.0
        self.scan_direction = 1
        self.scan_passes = 0
        self.scan_timer = QTimer(self)
        self.scan_timer.timeout.connect(self.update_scan_animation)
        self.sam_process = None
        self.sam_process_failure_handled = False

        self.image_view = ImageGraphicsView(self.image_scene)
        self.image_view.setMinimumSize(400, 350)
        self.image_view.setStyleSheet(
            "border: 1px solid #aaaaaa; background-color: #f4f4f4;"
        )

        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(phone_label)
        layout.addLayout(phone_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.phone_summary_label)
        layout.addWidget(self.preprocessing_toolbar)
        layout.addWidget(self.editor_toolbar)
        layout.addWidget(self.past_images_list)
        layout.addWidget(self.image_view, stretch=1)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # -------------------------------------------------------------------------
    # Image saving
    # -------------------------------------------------------------------------

    def select_and_save_image(self) -> None:
        """Select an image and open it in temporary upload pre-processing."""
        cleaned_phone_number = self.get_valid_phone_number()
        if cleaned_phone_number is None:
            return

        # Require a fresh lookup if the phone field changed after it was checked.
        if cleaned_phone_number != self.checked_phone_number:
            QMessageBox.warning(
                self, "Check Phone Number", "Check the phone number before adding an image."
            )
            return

        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
        )

        # An empty path means the user closed the file picker without choosing a file.
        if not image_path:
            return

        source_path = Path(image_path)
        pixmap = QPixmap(str(source_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Preview Failed", "The selected image cannot be shown.")
            return

        # Temporary image state stays only in memory until Confirm Image is clicked.
        self.temporary_image_source_path = source_path
        self.temporary_image_pixmap = pixmap
        self.preprocessing_mode = True
        self.show_preprocessing_preview()
        self.status_label.setText(
            "Image selected for pre-processing\n"
            "Rotate if needed, then click Confirm Image\n"
            f"{self.phone_number_status_text(cleaned_phone_number)}"
        )

    def show_preprocessing_preview(self) -> None:
        """Display the current temporary image without saving it."""
        if self.temporary_image_pixmap is None:
            return

        temporary_pixmap = self.temporary_image_pixmap
        temporary_source_path = self.temporary_image_source_path
        self.clear_image_scene()
        self.temporary_image_pixmap = temporary_pixmap
        self.temporary_image_source_path = temporary_source_path
        self.preprocessing_mode = True
        self.preprocessing_toolbar.setVisible(True)

        self.base_image_item = self.image_scene.addPixmap(temporary_pixmap)
        self.base_image_item.setZValue(0)
        self.image_scene.setSceneRect(self.base_image_item.boundingRect())
        self.image_view.fitInView(
            self.image_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def rotate_preview_left(self) -> None:
        """Rotate the temporary preview 90 degrees counterclockwise."""
        self.rotate_temporary_image(-90)

    def rotate_preview_right(self) -> None:
        """Rotate the temporary preview 90 degrees clockwise."""
        self.rotate_temporary_image(90)

    def rotate_temporary_image(self, degrees: int) -> None:
        """Rotate the in-memory image used by upload pre-processing."""
        if self.temporary_image_pixmap is None:
            return

        transform = QTransform().rotate(degrees)
        self.temporary_image_pixmap = self.temporary_image_pixmap.transformed(
            transform, Qt.TransformationMode.SmoothTransformation
        )
        self.show_preprocessing_preview()

    def confirm_preprocessed_image(self) -> None:
        """Save the confirmed preview and create one visualisation attempt."""
        cleaned_phone_number = self.get_valid_phone_number()
        if (
            cleaned_phone_number is None
            or cleaned_phone_number != self.checked_phone_number
            or self.temporary_image_pixmap is None
            or self.temporary_image_source_path is None
        ):
            return

        try:
            visualisation = self.create_visualisation_session(
                cleaned_phone_number, self.temporary_image_pixmap
            )
        except OSError as error:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"The image could not be saved.\n\n{error}",
            )
            return

        self.preprocessing_toolbar.setVisible(False)
        self.preprocessing_mode = False
        self.temporary_image_pixmap = None
        self.temporary_image_source_path = None
        self.current_visualisation_id = visualisation["id"]
        self.load_visualisations(cleaned_phone_number)
        self.refresh_phone_number_summary()
        saved_path = (
            self.data_folder / cleaned_phone_number / visualisation["image_path"]
        )
        self.status_label.setText(
            "Image saved successfully\nScanning room surfaces...\n"
            f"{self.phone_number_status_text(cleaned_phone_number)}\n"
            f"Saved file path: {saved_path}"
        )
        self.update_past_button()
        self.start_scan_animation(saved_path)
        self.run_sam31_segmentation(visualisation, saved_path)

    def cancel_preprocessing(self) -> None:
        """Discard the temporary image and return to the main screen."""
        self.temporary_image_pixmap = None
        self.temporary_image_source_path = None
        self.preprocessing_mode = False
        self.preprocessing_toolbar.setVisible(False)
        self.clear_image_scene()
        self.status_label.setText("Image pre-processing cancelled")

    def clear_form(self) -> None:
        """Clear the phone input, base image, segment overlays, and status message."""
        self.stop_active_sam_process()
        self.scan_timer.stop()
        self.cancel_preprocessing()
        self.close_editor_mode()
        self.phone_input.clear()
        self.clear_image_scene()
        self.reset_phone_lookup()
        self.status_label.clear()
        self.phone_input.setFocus()

    # -------------------------------------------------------------------------
    # Phone number cleaning and validation
    # -------------------------------------------------------------------------

    def clean_phone_number(self) -> str:
        """Normalize an Indian phone number to its ten-digit local form."""
        digits_only = re.sub(r"\D", "", self.phone_input.text().strip())

        # Indian country code 91 is removed only from a 12-digit number.
        if len(digits_only) == 12 and digits_only.startswith("91"):
            return digits_only[2:]

        return digits_only

    def get_valid_phone_number(self) -> str | None:
        """Normalize the entered phone number and return it only when valid."""
        normalized_phone_number = self.clean_phone_number()

        if re.fullmatch(r"\d{10}", normalized_phone_number):
            return normalized_phone_number

        QMessageBox.warning(
            self,
            "Invalid Phone Number",
            "Enter a valid 10-digit Indian phone number.\n\n"
            "A leading Indian country code 91 is also accepted.",
        )
        self.status_label.setText(
            "Invalid phone number. Enter exactly 10 digits, optionally with "
            "the Indian country code 91."
        )
        return None

    def phone_number_status_text(self, normalized_phone_number: str) -> str:
        """Return the original input and normalized number for status messages."""
        return (
            f"Original input: {self.phone_input.text().strip()}\n"
            f"Normalized phone number: {normalized_phone_number}"
        )

    def check_phone_number(self) -> None:
        """Look up a cleaned phone number in the local data folder."""
        cleaned_phone_number = self.get_valid_phone_number()
        if cleaned_phone_number is None:
            return

        self.checked_phone_number = cleaned_phone_number
        self.load_visualisations(cleaned_phone_number)
        self.upload_button.setEnabled(True)
        self.close_past_visualisations()
        self.refresh_phone_number_summary()

        # Existing-user flow: report all attempts and offer saved history when present.
        if self.visualisations:
            self.status_label.setText(
                f"Existing phone number found\n"
                f"{self.phone_number_status_text(cleaned_phone_number)}"
            )
        else:
            # New-user flow: only adding a new image is available.
            self.status_label.setText(
                f"New phone number\n"
                f"{self.phone_number_status_text(cleaned_phone_number)}\n"
                "No saved visualisations found"
            )
        self.update_past_button()

    # -------------------------------------------------------------------------
    # Past visualisations
    # -------------------------------------------------------------------------

    def show_saved_visualisations(self) -> None:
        """Show a visible list containing only saved visualisations."""
        self.past_images_list.clear()

        saved_visualisations = [
            visualisation
            for visualisation in self.visualisations
            if visualisation.get("is_saved") is True
        ]

        if not saved_visualisations:
            self.status_label.setText("No saved visualisations found")
            self.past_images_list.setVisible(False)
            self.close_past_button.setVisible(False)
            return

        for visualisation in reversed(saved_visualisations):
            visualisation_id = visualisation.get("id", "Unknown visualisation")
            created_at = visualisation.get("created_at", "Unknown date")
            image_name = Path(visualisation.get("image_path", "")).name
            folder_name = Path(visualisation.get("folder", "")).name
            display_name = image_name or folder_name or "No image name"
            item = QListWidgetItem(
                f"{visualisation_id} | {created_at} | {display_name}"
            )
            item.setData(Qt.ItemDataRole.UserRole, visualisation_id)
            self.past_images_list.addItem(item)

        self.past_images_list.setVisible(True)
        self.close_past_button.setVisible(True)
        self.status_label.setText(
            f"Showing {len(saved_visualisations)} saved visualisation(s)"
        )

    def open_saved_visualisation_from_list(self, item: QListWidgetItem) -> None:
        """Open the clicked saved visualisation directly in editable editor mode."""
        visualisation_id = item.data(Qt.ItemDataRole.UserRole)
        self.load_visualisation_session(visualisation_id, open_editor=True)

    def close_saved_visualisations_panel(self) -> None:
        """Hide the past list without deleting files or clearing the shown image."""
        self.past_images_list.clear()
        self.past_images_list.setVisible(False)
        self.close_past_button.setVisible(False)

    # Compatibility names retained for existing button connections and code.
    def show_past_visualisations(self) -> None:
        self.show_saved_visualisations()

    def display_past_image(self, item: QListWidgetItem) -> None:
        self.open_saved_visualisation_from_list(item)

    def close_past_visualisations(self) -> None:
        self.close_saved_visualisations_panel()

    def load_mock_mask_visualisation(self) -> None:
        """Developer entry point for testing the converted SAM mask session."""
        mock_folder = (
            Path(__file__).resolve().parent
            / "segmentation_tests"
            / "output"
            / "app_visualisation_mock"
        )
        image_path = mock_folder / "original.png"
        segments_path = mock_folder / "segments.json"

        try:
            with segments_path.open("r", encoding="utf-8") as segments_file:
                segment_document = json.load(segments_file)
        except (OSError, json.JSONDecodeError) as error:
            QMessageBox.warning(
                self,
                "Mock Visualisation Missing",
                f"The mock visualisation could not be loaded.\n\n{error}",
            )
            return

        segments = segment_document.get("segments")
        if not image_path.exists() or not isinstance(segments, list):
            QMessageBox.warning(
                self,
                "Mock Visualisation Missing",
                "Run the SAM bridge converter before loading the mask mock.",
            )
            return

        self.show_image(
            image_path,
            segments,
            is_saved=False,
            open_editor=True,
            segments_path=segments_path,
            is_mock=True,
        )
        self.current_visualisation_id = segment_document.get(
            "visualisation_id", "test_sam31_conversion"
        )
        self.status_label.setText(
            "Developer mask visualisation loaded\n"
            "Select wall or ceiling masks to test colour editing"
        )

    def reset_phone_lookup(self) -> None:
        """Reset lookup actions when the user edits or clears the phone field."""
        self.scan_timer.stop()
        self.preprocessing_toolbar.setVisible(False)
        self.preprocessing_mode = False
        self.temporary_image_pixmap = None
        self.temporary_image_source_path = None
        self.editor_toolbar.setVisible(False)
        self.editor_mode = False
        self.checked_phone_number = None
        self.visualisations = []
        self.current_visualisation_id = None
        self.upload_button.setEnabled(False)
        self.save_visualisation_button.setVisible(False)
        self.view_past_button.setVisible(False)
        self.close_past_button.setVisible(False)
        self.phone_summary_label.clear()
        self.past_images_list.clear()
        self.past_images_list.setVisible(False)

    # -------------------------------------------------------------------------
    # Graphics scene and hover overlays
    # -------------------------------------------------------------------------

    def start_scan_animation(self, image_path: Path) -> None:
        """Show the uploaded image with a scanning line while external SAM runs."""
        # Upload pre-processing is complete before scanning starts. Future real
        # Crop support belongs in that earlier pre-processing step.
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Preview Failed", "The saved image cannot be shown.")
            return

        current_visualisation_id = self.current_visualisation_id
        self.clear_image_scene()
        self.current_visualisation_id = current_visualisation_id
        self.pending_image_path = image_path

        self.base_image_item = self.image_scene.addPixmap(pixmap)
        self.base_image_item.setZValue(0)
        self.image_scene.setSceneRect(self.base_image_item.boundingRect())

        self.scan_position = 0.0
        self.scan_direction = 1
        self.scan_passes = 0
        self.scan_line_item = self.image_scene.addLine(
            0,
            0,
            0,
            pixmap.height(),
            QPen(QColor("#00d9ff"), max(2, pixmap.width() / 250)),
        )
        self.scan_line_item.setZValue(2)
        self.image_view.fitInView(
            self.image_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self.scan_timer.start(20)

    def update_scan_animation(self) -> None:
        """Move the scan line left-to-right and right-to-left across the image."""
        if self.base_image_item is None or self.scan_line_item is None:
            self.scan_timer.stop()
            return

        width = self.base_image_item.pixmap().width()
        height = self.base_image_item.pixmap().height()
        self.scan_position += self.scan_direction * max(2, width / 55)

        if self.scan_position >= width:
            self.scan_position = width
            self.scan_direction = -1
            self.scan_passes += 1
        elif self.scan_position <= 0:
            self.scan_position = 0
            self.scan_direction = 1
            self.scan_passes += 1

        self.scan_line_item.setLine(
            self.scan_position, 0, self.scan_position, height
        )

        # The animation continues until the SAM subprocess finishes. It is not
        # used as a timer because real segmentation duration varies by image.

    def finish_scan_animation(self, detected_count: int) -> None:
        """Finish loading and open the newly generated editable mask pre-cuts."""
        self.scan_timer.stop()
        if self.pending_image_path is None:
            return

        visualisation_id = self.current_visualisation_id
        self.pending_image_path = None
        self.load_visualisation_session(visualisation_id, open_editor=True)
        self.status_label.setText(
            f"Detected {detected_count} wall/ceiling segment(s)\n"
            f"Loaded visualisation: {visualisation_id}\n"
            "Select one or more wall or ceiling segments to edit"
        )

    def run_sam31_segmentation(self, visualisation: dict, image_path: Path) -> None:
        """Run SAM 3.1 outside the lightweight main app using its own venv."""
        project_folder = Path(__file__).resolve().parent
        sam_python = project_folder / ".venv-sam31" / "Scripts" / "python.exe"
        sam_script = (
            project_folder
            / "segmentation_tests"
            / "test_sam31_wall_ceiling.py"
        )
        session_folder = (
            self.data_folder
            / self.checked_phone_number
            / visualisation["folder"]
        )

        if not sam_python.exists() or not sam_script.exists():
            self.handle_sam_segmentation_failure(
                "SAM 3.1 environment or segmentation script is missing."
            )
            return

        # QProcess keeps the UI responsive and lets the scan line keep moving.
        # SAM writes masks and app-compatible segments.json into this session.
        self.sam_process_failure_handled = False
        self.sam_process = QProcess(self)
        self.sam_process.setWorkingDirectory(str(project_folder))
        self.sam_process.setProgram(str(sam_python))
        self.sam_process.setArguments(
            [
                str(sam_script),
                "--input",
                str(image_path),
                "--output",
                str(session_folder),
            ]
        )
        self.sam_process.finished.connect(self.handle_sam_process_finished)
        self.sam_process.errorOccurred.connect(self.handle_sam_process_error)
        self.set_sam_busy(True)
        self.status_label.setText("Running SAM 3.1 wall/ceiling segmentation...")
        self.sam_process.start()

    def set_sam_busy(self, is_busy: bool) -> None:
        """Prevent changing the active customer/session while SAM is running."""
        self.phone_input.setEnabled(not is_busy)
        self.check_phone_button.setEnabled(not is_busy)
        self.upload_button.setEnabled(not is_busy and self.checked_phone_number is not None)
        self.clear_button.setEnabled(not is_busy)

    def stop_active_sam_process(self) -> None:
        """Stop an active external scan when the whole form is explicitly cleared."""
        if self.sam_process is None:
            return
        self.sam_process_failure_handled = True
        if self.sam_process.state() != QProcess.ProcessState.NotRunning:
            self.sam_process.kill()
        self.sam_process = None
        self.set_sam_busy(False)

    def handle_sam_process_error(self, process_error) -> None:
        """Report a SAM subprocess that could not be started."""
        if process_error == QProcess.ProcessError.FailedToStart:
            self.handle_sam_segmentation_failure(
                "The SAM 3.1 subprocess could not be started."
            )

    def handle_sam_process_finished(self, exit_code: int, exit_status) -> None:
        """Open SAM masks on success or preserve the failed attempt on error."""
        if self.sam_process_failure_handled or self.sam_process is None:
            return

        standard_output = bytes(self.sam_process.readAllStandardOutput()).decode(
            errors="replace"
        )
        standard_error = bytes(self.sam_process.readAllStandardError()).decode(
            errors="replace"
        )
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            self.handle_sam_segmentation_failure(standard_error or standard_output)
            return

        visualisation_id = self.current_visualisation_id
        segments = self.load_segments_for_visualisation(visualisation_id)
        if segments is None:
            self.handle_sam_segmentation_failure(
                "SAM finished but did not create a valid segments.json."
            )
            return

        self.finish_scan_animation(len(segments))
        self.set_sam_busy(False)
        self.sam_process = None

    def handle_sam_segmentation_failure(self, reason: str) -> None:
        """Keep the unsaved session and original image when segmentation fails."""
        if self.sam_process_failure_handled:
            return
        self.sam_process_failure_handled = True
        self.scan_timer.stop()
        self.pending_image_path = None
        if self.scan_line_item is not None and self.scan_line_item.scene() is not None:
            self.image_scene.removeItem(self.scan_line_item)
        self.scan_line_item = None

        reason = reason.strip() or "Unknown SAM 3.1 error."
        if "out of memory" in reason.lower() or "gpu memory" in reason.lower():
            message = "SAM ran out of GPU memory. Try using a smaller image."
        else:
            # Keep the status useful without flooding the window with a traceback.
            message = f"Segmentation failed: {reason[-2000:]}"

        self.editor_mode = False
        self.editor_toolbar.setVisible(False)
        self.save_visualisation_button.setVisible(False)
        self.status_label.setText(message)
        self.set_sam_busy(False)
        self.sam_process = None

    def show_image(
        self,
        image_path: Path,
        segment_data: list,
        is_saved: bool,
        open_editor: bool = False,
        segments_path: Path | None = None,
        is_mock: bool = False,
    ) -> None:
        """Display an image and draw polygon or mask visualisation segments."""
        pixmap = QPixmap(str(image_path))

        if pixmap.isNull():
            QMessageBox.warning(self, "Preview Failed", "The saved image cannot be shown.")
            return

        self.clear_image_scene()

        # Add the image first so it becomes the base layer in the scene.
        self.base_image_item = self.image_scene.addPixmap(pixmap)
        self.base_image_item.setZValue(0)
        self.image_scene.setSceneRect(self.base_image_item.boundingRect())

        # Store the active visualisation's original.png path and base pixels.
        # Every realistic segment tint reads lighting and shadows from this image.
        self.current_image_path = image_path
        self.current_original_image = pixmap.toImage().convertToFormat(
            QImage.Format.Format_ARGB32
        )
        self.current_segments_path = segments_path
        self.is_mock_visualisation = is_mock

        for segment in segment_data:
            shape_type = segment.get("shape_type")
            points = None
            mask_path_text = None
            mask_image = None

            # Polygon segments are retained for older visualisations.
            if shape_type == "polygon":
                points = segment.get("points", [])
                if len(points) < 3:
                    continue
                try:
                    polygon = QPolygonF(
                        [QPointF(float(x), float(y)) for x, y in points]
                    )
                except (TypeError, ValueError):
                    continue
                shape_path = QPainterPath()
                shape_path.addPolygon(polygon)

            # Real SAM segments use a PNG mask as hover, selection, and tint clip.
            elif shape_type == "mask":
                mask_path_text = segment.get("mask_path")
                if not mask_path_text:
                    continue
                mask_path = image_path.parent / mask_path_text
                mask_image = QImage(str(mask_path))
                if mask_image.isNull():
                    self.status_label.setText(f"Mask could not be loaded: {mask_path}")
                    continue
                if mask_image.size() != self.current_original_image.size():
                    mask_image = mask_image.scaled(
                        self.current_original_image.size(),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.FastTransformation,
                    )
                shape_path = self.create_path_from_mask(mask_image)
                if shape_path.isEmpty():
                    continue
            else:
                continue

            segment_item = HoverSegmentItem(
                segment.get("id", "segment"),
                segment.get("type", "wall"),
                shape_type,
                shape_path,
                self.update_editor_toolbar_state,
                self.current_original_image,
                points=points,
                mask_path=mask_path_text,
                mask_image=mask_image,
                applied_colour=segment.get("applied_colour"),
            )
            self.image_scene.addItem(segment_item)
            self.segment_items.append(segment_item)

        self.image_view.fitInView(
            self.image_scene.sceneRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self.save_visualisation_button.setVisible(not is_mock)
        self.save_visualisation_button.setEnabled(not is_saved and not is_mock)
        self.editor_mode = open_editor
        self.editor_toolbar.setVisible(open_editor)
        self.update_editor_toolbar_state()

    def create_path_from_mask(self, mask_image: QImage) -> QPainterPath:
        """Convert white mask runs into one selectable QPainterPath.

        The path is used for hit testing, hover highlighting, selection outlines,
        and clipping realistic colour rendering to the exact SAM mask.
        """
        grayscale_mask = mask_image.convertToFormat(QImage.Format.Format_Grayscale8)
        path = QPainterPath()

        for y in range(grayscale_mask.height()):
            run_start = None
            for x in range(grayscale_mask.width()):
                is_mask_pixel = grayscale_mask.pixelColor(x, y).red() >= 128
                if is_mask_pixel and run_start is None:
                    run_start = x
                elif not is_mask_pixel and run_start is not None:
                    path.addRect(run_start, y, x - run_start, 1)
                    run_start = None
            if run_start is not None:
                path.addRect(run_start, y, grayscale_mask.width() - run_start, 1)

        return path

    def update_editor_toolbar_state(self) -> None:
        """Enable selection actions when at least one room segment is selected."""
        has_selection = any(item.is_selected for item in self.segment_items)
        editing_enabled = self.editor_mode and has_selection
        self.change_colour_button.setEnabled(editing_enabled)
        self.remove_colour_button.setEnabled(editing_enabled)
        self.undo_button.setEnabled(self.editor_mode and bool(self.editor_history))
        self.redo_button.setEnabled(self.editor_mode and bool(self.redo_history))

    def segment_colour_snapshot(self) -> dict:
        """Capture current segment colours for the simple action history."""
        return {
            item.segment_id: item.applied_colour
            for item in self.segment_items
        }

    def restore_segment_colour_snapshot(self, snapshot: dict) -> None:
        """Restore segment colours from an Undo or Redo snapshot."""
        for item in self.segment_items:
            item.set_applied_colour(snapshot.get(item.segment_id))
        self.update_editor_toolbar_state()

    def record_editor_action(self, action_name: str) -> None:
        """Record an editor action and segment state before it changes anything."""
        self.editor_history.append(
            {
                "action": action_name,
                "colours": self.segment_colour_snapshot(),
                "segments": [item.state() for item in self.segment_items],
            }
        )
        self.redo_history.clear()

    def change_selected_colour(self) -> None:
        """Apply the entered hex colour to every selected segment."""
        colour = self.hex_colour_input.text().strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
            QMessageBox.warning(
                self,
                "Invalid Colour",
                "Enter a six-digit hex colour such as #C8A27A.",
            )
            return

        selected_items = [item for item in self.segment_items if item.is_selected]
        if not selected_items:
            return

        self.record_editor_action("change_colour")
        for item in selected_items:
            item.set_applied_colour(colour.upper())
        self.save_current_segments_state()
        self.update_editor_toolbar_state()

    def remove_selected_colour(self) -> None:
        """Remove colour only from the currently selected segments."""
        selected_items = [item for item in self.segment_items if item.is_selected]
        if not selected_items:
            return

        self.record_editor_action("remove_colour")
        for item in selected_items:
            item.set_applied_colour(None)
        self.save_current_segments_state()
        self.update_editor_toolbar_state()

    def undo_editor_action(self) -> None:
        """Restore the previous recorded segment-colour state."""
        if not self.editor_history:
            return

        action = self.editor_history.pop()
        self.redo_history.append(
            {
                "action": action["action"],
                "colours": self.segment_colour_snapshot(),
                "segments": [item.state() for item in self.segment_items],
            }
        )
        self.restore_segment_colour_snapshot(action["colours"])
        self.save_current_segments_state()

    def redo_editor_action(self) -> None:
        """Reapply the most recently undone segment-colour state."""
        if not self.redo_history:
            return

        action = self.redo_history.pop()
        self.editor_history.append(
            {
                "action": action["action"],
                "colours": self.segment_colour_snapshot(),
                "segments": [item.state() for item in self.segment_items],
            }
        )
        self.restore_segment_colour_snapshot(action["colours"])
        self.save_current_segments_state()

    def close_editor_mode(self) -> None:
        """Exit editor mode while keeping the current visualisation displayed."""
        self.editor_mode = False
        self.editor_toolbar.setVisible(False)
        for item in self.segment_items:
            item.is_selected = False
            item.refresh_appearance()
        self.status_label.setText("Editor closed")

    def clear_image_scene(self) -> None:
        """Remove the base image and all hoverable segment overlays."""
        self.image_scene.clear()
        self.image_scene.setSceneRect(QRectF())
        self.base_image_item = None
        self.current_image_path = None
        self.current_original_image = None
        self.current_segments_path = None
        self.is_mock_visualisation = False
        self.segment_items.clear()
        self.current_visualisation_id = None
        self.scan_line_item = None
        self.pending_image_path = None
        self.temporary_image_pixmap = None
        self.temporary_image_source_path = None
        self.preprocessing_mode = False
        self.preprocessing_toolbar.setVisible(False)
        self.editor_history.clear()
        self.redo_history.clear()
        self.editor_mode = False
        self.editor_toolbar.setVisible(False)
        self.save_visualisation_button.setVisible(False)

    # -------------------------------------------------------------------------
    # Metadata loading and saving
    # -------------------------------------------------------------------------

    def create_visualisation_session(
        self, cleaned_phone_number: str, confirmed_pixmap: QPixmap
    ) -> dict:
        """Create one visualisation folder, original image, and index entry."""
        now = datetime.now()
        base_visualisation_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        visualisation_id = base_visualisation_id
        suffix = 1

        # Keep every attempt separate even if two are confirmed in the same second.
        phone_folder = self.data_folder / cleaned_phone_number
        while (phone_folder / "visualisations" / visualisation_id).exists():
            visualisation_id = f"{base_visualisation_id}_{suffix:02d}"
            suffix += 1

        relative_folder = Path("visualisations") / visualisation_id
        session_folder = phone_folder / relative_folder
        session_folder.mkdir(parents=True, exist_ok=True)

        image_path = session_folder / "original.png"

        if not confirmed_pixmap.save(str(image_path), "PNG"):
            raise OSError("The confirmed image could not be saved as original.png.")

        visualisation = {
            "id": visualisation_id,
            "folder": relative_folder.as_posix(),
            "image_path": (relative_folder / "original.png").as_posix(),
            "segments_path": (relative_folder / "segments.json").as_posix(),
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "is_saved": False,
        }

        # metadata.json is the phone-number-level index of all attempts. SAM is
        # run after this write, so a failed scan still remains a visible attempt
        # with its confirmed original.png and is_saved false.
        metadata = self.load_metadata(cleaned_phone_number)
        metadata["visualisations"].append(visualisation)
        self.write_metadata(cleaned_phone_number, metadata)
        return visualisation

    def create_demo_segment_data(self, width: int, height: int) -> list:
        """Create fake wall and ceiling pre-cuts for one new visualisation."""
        definitions = [
            ("segment_001", "wall", [(0.10, 0.18), (0.25, 0.10), (0.42, 0.22), (0.35, 0.40), (0.16, 0.36)]),
            ("segment_002", "wall", [(0.57, 0.16), (0.82, 0.20), (0.88, 0.39), (0.71, 0.49), (0.54, 0.34)]),
            ("segment_003", "wall", [(0.26, 0.59), (0.48, 0.52), (0.69, 0.66), (0.60, 0.86), (0.34, 0.82)]),
            ("segment_004", "ceiling", [(0.08, 0.08), (0.46, 0.04), (0.38, 0.18), (0.12, 0.22)]),
            ("segment_005", "ceiling", [(0.48, 0.04), (0.92, 0.10), (0.84, 0.23), (0.55, 0.17)]),
        ]

        return [
            {
                "id": segment_id,
                "type": segment_type,
                "shape_type": "polygon",
                "points": [
                    [round(width * x), round(height * y)]
                    for x, y in proportional_points
                ],
                "applied_colour": None,
            }
            for segment_id, segment_type, proportional_points in definitions
        ]

    def load_visualisation_session(
        self, visualisation_id: str, open_editor: bool = False
    ) -> None:
        """Load one visualisation's own image and saved segment state."""
        visualisation = self.find_visualisation(visualisation_id)
        if visualisation is None or self.checked_phone_number is None:
            self.status_label.setText(
                f"Saved visualisation could not be found: {visualisation_id}"
            )
            return

        image_relative_path = visualisation.get("image_path")
        if not image_relative_path:
            self.status_label.setText("This older visualisation has no image path.")
            return

        image_path = self.data_folder / self.checked_phone_number / image_relative_path
        if not image_path.exists():
            self.status_label.setText(
                f"This saved visualisation's original image is missing: {image_path}"
            )
            return

        segment_data = self.load_segments_for_visualisation(visualisation_id)
        if segment_data is None:
            return
        segments_path = (
            self.data_folder
            / self.checked_phone_number
            / visualisation["segments_path"]
        )

        # Set the active session before opening the editor so colour edits always
        # write back to this exact existing visualisation, never a duplicate.
        self.current_visualisation_id = visualisation_id
        self.show_image(
            image_path,
            segment_data,
            is_saved=visualisation.get("is_saved") is True,
            open_editor=open_editor,
            segments_path=segments_path,
        )
        self.current_visualisation_id = visualisation_id
        if open_editor:
            self.close_saved_visualisations_panel()
        self.status_label.setText(
            f"Loaded saved visualisation: {visualisation_id}\n"
            f"Loaded {len(self.segment_items)} segment(s)"
        )

    def open_visualisation_editor(self, visualisation_id: str) -> None:
        """Open an existing saved visualisation in editor mode."""
        self.load_visualisation_session(visualisation_id, open_editor=True)

    def load_segments_for_visualisation(self, visualisation_id: str) -> list | None:
        """Read one visualisation's segments.json without regenerating pre-cuts."""
        visualisation = self.find_visualisation(visualisation_id)
        if visualisation is None or self.checked_phone_number is None:
            return None

        segments_relative_path = visualisation.get("segments_path")
        if not segments_relative_path:
            self.status_label.setText(
                "This older visualisation has no saved pre-cuts/segments."
            )
            return None

        segments_path = (
            self.data_folder / self.checked_phone_number / segments_relative_path
        )
        try:
            with segments_path.open("r", encoding="utf-8") as segments_file:
                segment_document = json.load(segments_file)
        except FileNotFoundError:
            self.status_label.setText(
                "This older visualisation has no saved pre-cuts/segments."
            )
            return None
        except (OSError, json.JSONDecodeError):
            self.status_label.setText(
                "This visualisation's segment data could not be loaded."
            )
            return None

        segments = segment_document.get("segments")
        if not isinstance(segments, list):
            self.status_label.setText("This visualisation has invalid segment data.")
            return None
        return segments

    def save_current_segments_state(self) -> bool:
        """Save current colour data and report successful existing-session edits."""
        saved = self.save_segments_for_current_visualisation()
        if saved and not self.is_mock_visualisation:
            self.status_label.setText("Updated saved visualisation colour data")
        return saved

    def save_segments_for_current_visualisation(self) -> bool:
        """Save polygon points or mask references and colours to segments.json."""
        if self.current_visualisation_id is None:
            return False

        segments_path = self.current_segments_path
        if segments_path is None:
            if self.checked_phone_number is None:
                return False
            visualisation = self.find_visualisation(self.current_visualisation_id)
            if visualisation is None or not visualisation.get("segments_path"):
                return False
            segments_path = (
                self.data_folder
                / self.checked_phone_number
                / visualisation["segments_path"]
            )

        segments = []
        for item in self.segment_items:
            segment_record = {
                "id": item.segment_id,
                "type": item.segment_type,
                "shape_type": item.shape_type,
                "applied_colour": item.applied_colour,
            }

            if item.shape_type == "mask":
                # Preserve the visualisation-relative mask reference exactly.
                segment_record["mask_path"] = item.mask_path
            else:
                segment_record["points"] = item.points

            segments.append(segment_record)

        try:
            with segments_path.open("w", encoding="utf-8") as segments_file:
                json.dump(
                    {
                        "visualisation_id": self.current_visualisation_id,
                        "segments": segments,
                },
                segments_file,
                indent=2,
            )
            return True
        except OSError as error:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"The segment colours could not be saved.\n\n{error}",
            )
            return False

    def load_metadata(self, cleaned_phone_number: str) -> dict:
        """Read metadata.json, the phone-number-level visualisation index."""
        metadata_path = self.data_folder / cleaned_phone_number / "metadata.json"
        empty_metadata = {
            "phone_number": cleaned_phone_number,
            "visualisations": [],
        }

        if not metadata_path.exists():
            return empty_metadata

        try:
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except (OSError, json.JSONDecodeError):
            return empty_metadata

        if not isinstance(metadata.get("visualisations"), list):
            return empty_metadata
        return metadata

    def write_metadata(self, cleaned_phone_number: str, metadata: dict) -> None:
        """Write all visualisation metadata to the phone number's JSON file."""
        phone_folder = self.data_folder / cleaned_phone_number
        phone_folder.mkdir(parents=True, exist_ok=True)
        metadata_path = phone_folder / "metadata.json"

        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def load_visualisations(self, cleaned_phone_number: str) -> None:
        """Load all visualisation attempts for the checked phone number."""
        metadata = self.load_metadata(cleaned_phone_number)
        self.visualisations = metadata["visualisations"]

    # -------------------------------------------------------------------------
    # Visualisation counting and saved state
    # -------------------------------------------------------------------------

    def find_visualisation(self, visualisation_id: str) -> dict | None:
        """Find one visualisation entry by its metadata ID."""
        return next(
            (
                visualisation
                for visualisation in self.visualisations
                if visualisation.get("id") == visualisation_id
            ),
            None,
        )

    def mark_current_visualisation_saved(self) -> None:
        """Mark the currently displayed visualisation as saved in metadata."""
        if self.checked_phone_number is None or self.current_visualisation_id is None:
            return

        metadata = self.load_metadata(self.checked_phone_number)
        for visualisation in metadata["visualisations"]:
            if visualisation.get("id") == self.current_visualisation_id:
                visualisation["is_saved"] = True
                break
        else:
            return

        try:
            self.write_metadata(self.checked_phone_number, metadata)
        except OSError as error:
            QMessageBox.critical(
                self, "Save Failed", f"The visualisation metadata could not be saved.\n\n{error}"
            )
            return

        self.load_visualisations(self.checked_phone_number)
        self.refresh_phone_number_summary()
        self.save_visualisation_button.setEnabled(False)
        self.update_past_button()
        self.status_label.setText("This visualisation has been saved")

    def visualisation_counts_text(self) -> str:
        """Return the total and saved visualisation counts for status messages."""
        total_count = len(self.visualisations)
        saved_count = sum(
            visualisation.get("is_saved") is True
            for visualisation in self.visualisations
        )
        return (
            f"Total visualisations done: {total_count}\n"
            f"Saved visualisations: {saved_count}"
        )

    def refresh_phone_number_summary(self) -> None:
        """Refresh the total and saved counts shown for the checked phone number."""
        if self.checked_phone_number is None:
            self.phone_summary_label.clear()
            return

        self.phone_summary_label.setText(self.visualisation_counts_text())

    def update_past_button(self) -> None:
        """Show the past button only when at least one saved visualisation exists."""
        has_saved_visualisations = any(
            visualisation.get("is_saved") is True
            for visualisation in self.visualisations
        )
        self.view_past_button.setVisible(has_saved_visualisations)


def main() -> None:
    """Start the desktop application."""
    app = QApplication(sys.argv)
    window = ImageUploadWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
