"""
Quantitative Temporal Module for MedRAG-X
==========================================

This module provides temporal analysis capabilities for tracking brain MRI changes
over time, computing progression metrics, and integrating temporal features into
the multimodal knowledge graph.

Author: MedRAG-X Team
Project: Extending RAG-Anything for MRI-Aware Clinical Reasoning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import nibabel as nib
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
import warnings

warnings.filterwarnings('ignore')


@dataclass
class TemporalScan:
    """Represents a single MRI scan with temporal metadata."""
    scan_id: str
    patient_id: str
    acquisition_date: datetime
    modalities: Dict[str, np.ndarray]  # e.g., {'T1': array, 'T2': array, 'FLAIR': array}
    segmentation: Optional[np.ndarray] = None
    clinical_notes: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class TemporalMetrics:
    """Container for computed temporal metrics."""
    patient_id: str
    timepoint_1: datetime
    timepoint_2: datetime
    time_delta_days: float

    # Volume changes
    total_volume_change: float
    enhancing_volume_change: float
    necrotic_volume_change: float
    edema_volume_change: float

    # Growth rates (mm³/day)
    total_growth_rate: float
    enhancing_growth_rate: float

    # Shape changes
    hausdorff_distance: float
    centroid_displacement: float
    surface_area_change: float

    # Intensity changes
    mean_intensity_change: Dict[str, float]
    intensity_heterogeneity_change: float

    # Clinical significance flags
    rapid_progression: bool
    new_lesions_detected: bool
    treatment_response: str  # 'progressive', 'stable', 'responsive'

    # Additional metrics
    additional_metrics: Dict = field(default_factory=dict)


class VolumetricAnalyzer:
    """Analyzes volumetric changes in tumor regions."""

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Args:
            voxel_spacing: Voxel dimensions in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing
        self.voxel_volume = np.prod(voxel_spacing)

    def compute_volume(self, segmentation: np.ndarray, label: int = 1) -> float:
        """
        Compute volume for a specific segmentation label.

        Args:
            segmentation: 3D segmentation mask
            label: Label value to compute volume for

        Returns:
            Volume in mm³
        """
        voxel_count = np.sum(segmentation == label)
        return voxel_count * self.voxel_volume

    def compute_all_volumes(self, segmentation: np.ndarray) -> Dict[str, float]:
        """
        Compute volumes for all tumor components.
        BraTS labels: 1=necrotic, 2=edema, 4=enhancing

        Returns:
            Dictionary of volumes for each component
        """
        volumes = {
            'necrotic': self.compute_volume(segmentation, label=1),
            'edema': self.compute_volume(segmentation, label=2),
            'enhancing': self.compute_volume(segmentation, label=4),
        }
        volumes['total'] = sum(volumes.values())
        volumes['core'] = volumes['necrotic'] + volumes['enhancing']  # Tumor core
        return volumes

    def compute_volume_change(self, seg_t1: np.ndarray, seg_t2: np.ndarray) -> Dict[str, float]:
        """
        Compute volumetric changes between two timepoints.

        Returns:
            Dictionary of volume changes (absolute and percentage)
        """
        vol_t1 = self.compute_all_volumes(seg_t1)
        vol_t2 = self.compute_all_volumes(seg_t2)

        changes = {}
        for component in vol_t1.keys():
            abs_change = vol_t2[component] - vol_t1[component]
            pct_change = (abs_change / vol_t1[component] * 100) if vol_t1[component] > 0 else 0
            changes[f'{component}_absolute'] = abs_change
            changes[f'{component}_percentage'] = pct_change

        return changes


class MorphologicalAnalyzer:
    """Analyzes morphological and shape changes."""

    @staticmethod
    def compute_centroid(segmentation: np.ndarray, label: int = 1) -> np.ndarray:
        """Compute centroid of segmentation region."""
        coords = np.argwhere(segmentation == label)
        if len(coords) == 0:
            return np.array([0, 0, 0])
        return coords.mean(axis=0)

    def compute_centroid_displacement(self, seg_t1: np.ndarray, seg_t2: np.ndarray,
                                      voxel_spacing: Tuple[float, float, float],
                                      label: int = 4) -> float:
        """
        Compute displacement of tumor centroid between timepoints.

        Args:
            label: Tumor component to track (default: 4 = enhancing)

        Returns:
            Displacement in mm
        """
        c1 = self.compute_centroid(seg_t1, label)
        c2 = self.compute_centroid(seg_t2, label)
        displacement_voxels = np.linalg.norm(c2 - c1)
        # Convert to mm using average voxel spacing
        displacement_mm = displacement_voxels * np.mean(voxel_spacing)
        return displacement_mm

    def compute_hausdorff_distance(self, seg_t1: np.ndarray, seg_t2: np.ndarray,
                                   label: int = 4) -> float:
        """
        Compute Hausdorff distance between tumor surfaces.
        Measures maximum shape change.
        """
        coords_t1 = np.argwhere(seg_t1 == label)
        coords_t2 = np.argwhere(seg_t2 == label)

        if len(coords_t1) == 0 or len(coords_t2) == 0:
            return 0.0

        # Sample points if too many (for computational efficiency)
        max_points = 1000
        if len(coords_t1) > max_points:
            indices = np.random.choice(len(coords_t1), max_points, replace=False)
            coords_t1 = coords_t1[indices]
        if len(coords_t2) > max_points:
            indices = np.random.choice(len(coords_t2), max_points, replace=False)
            coords_t2 = coords_t2[indices]

        dist_forward = directed_hausdorff(coords_t1, coords_t2)[0]
        dist_backward = directed_hausdorff(coords_t2, coords_t1)[0]

        return max(dist_forward, dist_backward)

    def compute_surface_area(self, segmentation: np.ndarray, label: int = 1,
                             voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
        """
        Estimate surface area using marching cubes approach.
        """
        binary_mask = (segmentation == label).astype(np.uint8)

        # Simple surface estimation: count boundary voxels
        # Dilate and subtract to find surface
        dilated = ndimage.binary_dilation(binary_mask)
        surface = dilated.astype(int) - binary_mask.astype(int)
        surface_voxels = np.sum(surface)

        # Approximate surface area (this is a simplification)
        voxel_face_area = np.mean([voxel_spacing[i] * voxel_spacing[j]
                                   for i in range(3) for j in range(i + 1, 3)])
        return surface_voxels * voxel_face_area * 2  # *2 for rough calibration


class IntensityAnalyzer:
    """Analyzes intensity changes in MRI sequences."""

    def compute_mean_intensity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Compute mean intensity within masked region."""
        masked_values = image[mask > 0]
        return np.mean(masked_values) if len(masked_values) > 0 else 0.0

    def compute_intensity_heterogeneity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute intensity heterogeneity (coefficient of variation).
        Higher values indicate more heterogeneous tumor.
        """
        masked_values = image[mask > 0]
        if len(masked_values) == 0:
            return 0.0
        mean_val = np.mean(masked_values)
        std_val = np.std(masked_values)
        return (std_val / mean_val) if mean_val > 0 else 0.0

    def compute_intensity_changes(self, img_t1: np.ndarray, img_t2: np.ndarray,
                                  seg_t1: np.ndarray, seg_t2: np.ndarray,
                                  modality: str) -> Dict[str, float]:
        """
        Compute intensity-based changes for a specific modality.
        """
        # Use intersection of masks for fair comparison
        common_mask = (seg_t1 > 0) & (seg_t2 > 0)

        mean_t1 = self.compute_mean_intensity(img_t1, common_mask)
        mean_t2 = self.compute_mean_intensity(img_t2, common_mask)

        het_t1 = self.compute_intensity_heterogeneity(img_t1, seg_t1)
        het_t2 = self.compute_intensity_heterogeneity(img_t2, seg_t2)

        return {
            f'{modality}_mean_change': mean_t2 - mean_t1,
            f'{modality}_mean_change_pct': ((mean_t2 - mean_t1) / mean_t1 * 100) if mean_t1 > 0 else 0,
            f'{modality}_heterogeneity_t1': het_t1,
            f'{modality}_heterogeneity_t2': het_t2,
            f'{modality}_heterogeneity_change': het_t2 - het_t1,
        }


class LesionTracker:
    """Tracks individual lesions across timepoints."""

    def __init__(self, min_volume_threshold: float = 50.0):
        """
        Args:
            min_volume_threshold: Minimum volume (mm³) to consider a lesion
        """
        self.min_volume_threshold = min_volume_threshold

    def detect_lesions(self, segmentation: np.ndarray,
                       voxel_volume: float) -> List[Dict]:
        """
        Detect and characterize individual lesions.

        Returns:
            List of lesion dictionaries with properties
        """
        # Label connected components
        labeled_array, num_features = ndimage.label(segmentation > 0)

        lesions = []
        for label_id in range(1, num_features + 1):
            lesion_mask = (labeled_array == label_id)
            volume = np.sum(lesion_mask) * voxel_volume

            if volume < self.min_volume_threshold:
                continue

            centroid = np.argwhere(lesion_mask).mean(axis=0)

            lesions.append({
                'id': label_id,
                'volume': volume,
                'centroid': centroid.tolist(),
                'voxel_count': np.sum(lesion_mask),
                'bounding_box': self._get_bounding_box(lesion_mask),
            })

        return lesions

    def _get_bounding_box(self, mask: np.ndarray) -> Dict:
        """Get bounding box coordinates of lesion."""
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return {'min': [0, 0, 0], 'max': [0, 0, 0]}

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        return {
            'min': min_coords.tolist(),
            'max': max_coords.tolist(),
        }

    def track_lesions(self, lesions_t1: List[Dict], lesions_t2: List[Dict],
                      max_distance: float = 20.0) -> Dict:
        """
        Track lesions between timepoints using centroid matching.

        Args:
            max_distance: Maximum centroid distance (mm) to match lesions

        Returns:
            Tracking results with matched, disappeared, and new lesions
        """
        matched = []
        new_lesions = []
        disappeared = []

        # Create distance matrix
        if len(lesions_t1) > 0 and len(lesions_t2) > 0:
            centroids_t1 = np.array([l['centroid'] for l in lesions_t1])
            centroids_t2 = np.array([l['centroid'] for l in lesions_t2])

            # Compute pairwise distances
            distances = np.sqrt(((centroids_t1[:, None] - centroids_t2[None, :]) ** 2).sum(axis=2))

            # Greedy matching
            used_t2 = set()
            for i, lesion_t1 in enumerate(lesions_t1):
                min_dist_idx = distances[i].argmin()
                min_dist = distances[i, min_dist_idx]

                if min_dist < max_distance and min_dist_idx not in used_t2:
                    matched.append({
                        'lesion_t1': lesion_t1,
                        'lesion_t2': lesions_t2[min_dist_idx],
                        'distance': float(min_dist),
                        'volume_change': lesions_t2[min_dist_idx]['volume'] - lesion_t1['volume'],
                    })
                    used_t2.add(min_dist_idx)
                else:
                    disappeared.append(lesion_t1)

            # Find new lesions
            for j, lesion_t2 in enumerate(lesions_t2):
                if j not in used_t2:
                    new_lesions.append(lesion_t2)
        else:
            disappeared = lesions_t1
            new_lesions = lesions_t2

        return {
            'matched': matched,
            'new': new_lesions,
            'disappeared': disappeared,
            'n_matched': len(matched),
            'n_new': len(new_lesions),
            'n_disappeared': len(disappeared),
        }


class ProgressionClassifier:
    """Classifies disease progression based on temporal metrics."""

    def __init__(self):
        # RANO criteria thresholds (Response Assessment in Neuro-Oncology)
        self.progression_threshold = 25.0  # % increase
        self.response_threshold = -50.0  # % decrease
        self.rapid_growth_rate = 100.0  # mm³/day

    def classify_response(self, volume_change_pct: float,
                          new_lesions: bool) -> str:
        """
        Classify treatment response based on RANO-like criteria.

        Args:
            volume_change_pct: Percentage change in tumor volume
            new_lesions: Whether new lesions were detected

        Returns:
            Classification: 'progressive', 'stable', 'partial_response', 'complete_response'
        """
        if new_lesions or volume_change_pct >= self.progression_threshold:
            return 'progressive'
        elif volume_change_pct <= self.response_threshold:
            return 'partial_response'
        elif abs(volume_change_pct) < 10.0:  # Less than 10% change
            return 'stable'
        else:
            return 'minimal_change'

    def assess_clinical_significance(self, metrics: Dict) -> Dict[str, bool]:
        """
        Assess clinical significance of changes.

        Returns:
            Dictionary of clinical flags
        """
        flags = {
            'rapid_progression': False,
            'new_lesions_detected': False,
            'significant_volume_increase': False,
            'significant_volume_decrease': False,
            'infiltrative_growth': False,
            'central_necrosis_increase': False,
        }

        # Rapid progression
        if metrics.get('total_growth_rate', 0) > self.rapid_growth_rate:
            flags['rapid_progression'] = True

        # New lesions
        if metrics.get('n_new_lesions', 0) > 0:
            flags['new_lesions_detected'] = True

        # Volume changes
        total_change = metrics.get('total_percentage', 0)
        if total_change >= self.progression_threshold:
            flags['significant_volume_increase'] = True
        elif total_change <= self.response_threshold:
            flags['significant_volume_decrease'] = True

        # Infiltrative growth (large Hausdorff distance relative to volume change)
        hausdorff = metrics.get('hausdorff_distance', 0)
        if hausdorff > 15.0 and total_change < 50.0:  # Shape change without proportional volume change
            flags['infiltrative_growth'] = True

        # Necrotic core expansion
        necrotic_change = metrics.get('necrotic_percentage', 0)
        if necrotic_change > 30.0:
            flags['central_necrosis_increase'] = True

        return flags


class TemporalFeatureExtractor(nn.Module):
    """Neural network for extracting temporal features from paired scans."""

    def __init__(self, input_channels: int = 4, feature_dim: int = 256):
        """
        Args:
            input_channels: Number of input modalities (e.g., 4 for T1, T2, FLAIR, T1ce)
            feature_dim: Dimension of output feature vector
        """
        super().__init__()

        # Siamese-like encoder for each timepoint
        self.encoder = nn.Sequential(
            self._make_conv_block(input_channels, 32),
            self._make_conv_block(32, 64),
            self._make_conv_block(64, 128),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4 * 4, feature_dim),
            nn.ReLU(),
        )

        # Temporal fusion network
        self.temporal_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
        )

        # Change prediction head
        self.change_predictor = nn.Sequential(
            nn.Linear(feature_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Predict: volume change, progression risk, treatment response
        )

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a 3D convolutional block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, scan_t1: torch.Tensor, scan_t2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            scan_t1: First timepoint [batch, channels, D, H, W]
            scan_t2: Second timepoint [batch, channels, D, H, W]

        Returns:
            Dictionary with temporal features and predictions
        """
        # Encode both timepoints
        features_t1 = self.encoder(scan_t1)
        features_t2 = self.encoder(scan_t2)

        # Concatenate and fuse
        combined = torch.cat([features_t1, features_t2], dim=1)
        temporal_features = self.temporal_fusion(combined)

        # Predict changes
        predictions = self.change_predictor(temporal_features)

        return {
            'features_t1': features_t1,
            'features_t2': features_t2,
            'temporal_features': temporal_features,
            'volume_change_pred': predictions[:, 0],
            'progression_risk': torch.sigmoid(predictions[:, 1]),
            'response_score': torch.tanh(predictions[:, 2]),
        }


class QuantitativeTemporalModule:
    """
    Main module coordinating all temporal analysis components.
    """

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize the Quantitative Temporal Module.

        Args:
            voxel_spacing: Voxel dimensions in mm (x, y, z)
        """
        self.voxel_spacing = voxel_spacing

        # Initialize sub-modules
        self.volumetric_analyzer = VolumetricAnalyzer(voxel_spacing)
        self.morphological_analyzer = MorphologicalAnalyzer()
        self.intensity_analyzer = IntensityAnalyzer()
        self.lesion_tracker = LesionTracker()
        self.progression_classifier = ProgressionClassifier()

        # Neural feature extractor (optional, requires training)
        self.feature_extractor = None

    def analyze_temporal_pair(self, scan_t1: TemporalScan,
                              scan_t2: TemporalScan) -> TemporalMetrics:
        """
        Comprehensive temporal analysis between two timepoints.

        Args:
            scan_t1: Earlier scan
            scan_t2: Later scan

        Returns:
            TemporalMetrics object with all computed metrics
        """
        # Validate inputs
        assert scan_t1.patient_id == scan_t2.patient_id, "Scans must be from same patient"
        assert scan_t1.segmentation is not None and scan_t2.segmentation is not None, \
            "Both scans must have segmentations"

        # Calculate time difference
        time_delta = scan_t2.acquisition_date - scan_t1.acquisition_date
        time_delta_days = time_delta.total_seconds() / (24 * 3600)

        # 1. Volumetric analysis
        volume_changes = self.volumetric_analyzer.compute_volume_change(
            scan_t1.segmentation, scan_t2.segmentation
        )

        # Growth rates
        total_growth_rate = volume_changes['total_absolute'] / time_delta_days if time_delta_days > 0 else 0
        enhancing_growth_rate = volume_changes['enhancing_absolute'] / time_delta_days if time_delta_days > 0 else 0

        # 2. Morphological analysis
        hausdorff_dist = self.morphological_analyzer.compute_hausdorff_distance(
            scan_t1.segmentation, scan_t2.segmentation
        )

        centroid_disp = self.morphological_analyzer.compute_centroid_displacement(
            scan_t1.segmentation, scan_t2.segmentation, self.voxel_spacing
        )

        surface_area_t1 = self.morphological_analyzer.compute_surface_area(
            scan_t1.segmentation, label=4, voxel_spacing=self.voxel_spacing
        )
        surface_area_t2 = self.morphological_analyzer.compute_surface_area(
            scan_t2.segmentation, label=4, voxel_spacing=self.voxel_spacing
        )
        surface_area_change = surface_area_t2 - surface_area_t1

        # 3. Intensity analysis (for available modalities)
        intensity_changes = {}
        for modality in scan_t1.modalities.keys():
            if modality in scan_t2.modalities:
                changes = self.intensity_analyzer.compute_intensity_changes(
                    scan_t1.modalities[modality],
                    scan_t2.modalities[modality],
                    scan_t1.segmentation,
                    scan_t2.segmentation,
                    modality
                )
                intensity_changes.update(changes)

        # Mean intensity change across modalities
        mean_intensity_change = {
            modality: intensity_changes.get(f'{modality}_mean_change', 0)
            for modality in scan_t1.modalities.keys()
        }

        # Heterogeneity change (using first available modality)
        het_key = f'{list(scan_t1.modalities.keys())[0]}_heterogeneity_change'
        intensity_het_change = intensity_changes.get(het_key, 0)

        # 4. Lesion tracking
        voxel_volume = self.volumetric_analyzer.voxel_volume
        lesions_t1 = self.lesion_tracker.detect_lesions(scan_t1.segmentation, voxel_volume)
        lesions_t2 = self.lesion_tracker.detect_lesions(scan_t2.segmentation, voxel_volume)
        tracking_results = self.lesion_tracker.track_lesions(lesions_t1, lesions_t2)

        # 5. Progression classification
        treatment_response = self.progression_classifier.classify_response(
            volume_changes['total_percentage'],
            tracking_results['n_new'] > 0
        )

        # Clinical significance assessment
        metrics_for_assessment = {
            'total_growth_rate': total_growth_rate,
            'n_new_lesions': tracking_results['n_new'],
            'total_percentage': volume_changes['total_percentage'],
            'hausdorff_distance': hausdorff_dist,
            'necrotic_percentage': volume_changes['necrotic_percentage'],
        }
        clinical_flags = self.progression_classifier.assess_clinical_significance(
            metrics_for_assessment
        )

        # Compile results
        metrics = TemporalMetrics(
            patient_id=scan_t1.patient_id,
            timepoint_1=scan_t1.acquisition_date,
            timepoint_2=scan_t2.acquisition_date,
            time_delta_days=time_delta_days,

            # Volume changes
            total_volume_change=volume_changes['total_absolute'],
            enhancing_volume_change=volume_changes['enhancing_absolute'],
            necrotic_volume_change=volume_changes['necrotic_absolute'],
            edema_volume_change=volume_changes['edema_absolute'],

            # Growth rates
            total_growth_rate=total_growth_rate,
            enhancing_growth_rate=enhancing_growth_rate,

            # Shape changes
            hausdorff_distance=hausdorff_dist,
            centroid_displacement=centroid_disp,
            surface_area_change=surface_area_change,

            # Intensity changes
            mean_intensity_change=mean_intensity_change,
            intensity_heterogeneity_change=intensity_het_change,

            # Clinical assessment
            rapid_progression=clinical_flags['rapid_progression'],
            new_lesions_detected=clinical_flags['new_lesions_detected'],
            treatment_response=treatment_response,

            # Additional metrics
            additional_metrics={
                'volume_changes': volume_changes,
                'intensity_changes': intensity_changes,
                'lesion_tracking': tracking_results,
                'clinical_flags': clinical_flags,
                'surface_area_t1': surface_area_t1,
                'surface_area_t2': surface_area_t2,
            }
        )

        return metrics

    def generate_temporal_report(self, metrics: TemporalMetrics) -> str:
        """
        Generate a human-readable clinical report from temporal metrics.

        Args:
            metrics: Computed temporal metrics

        Returns:
            Formatted clinical report string
        """
        report = f"""
TEMPORAL ANALYSIS REPORT
========================
Patient ID: {metrics.patient_id}
Baseline Scan: {metrics.timepoint_1.strftime('%Y-%m-%d')}
Follow-up Scan: {metrics.timepoint_2.strftime('%Y-%m-%d')}
Time Interval: {metrics.time_delta_days:.1f} days ({metrics.time_delta_days / 30:.1f} months)

VOLUMETRIC CHANGES
------------------
Total Tumor Volume Change: {metrics.total_volume_change:+.1f} mm³
Enhancing Component: {metrics.enhancing_volume_change:+.1f} mm³
Necrotic Component: {metrics.necrotic_volume_change:+.1f} mm³
Edema: {metrics.edema_volume_change:+.1f} mm³

GROWTH DYNAMICS
---------------
Total Growth Rate: {metrics.total_growth_rate:+.2f} mm³/day
Enhancing Growth Rate: {metrics.enhancing_growth_rate:+.2f} mm³/day

MORPHOLOGICAL CHANGES
---------------------
Hausdorff Distance: {metrics.hausdorff_distance:.2f} mm
Centroid Displacement: {metrics.centroid_displacement:.2f} mm
Surface Area Change: {metrics.surface_area_change:+.1f} mm²

INTENSITY PATTERNS
------------------
"""
        for modality, change in metrics.mean_intensity_change.items():
            report += f"{modality} Mean Intensity Change: {change:+.2f}\n"

        report += f"Heterogeneity Change: {metrics.intensity_heterogeneity_change:+.4f}\n"

        report += f"""
CLINICAL ASSESSMENT
-------------------
Treatment Response: {metrics.treatment_response.upper()}
Rapid Progression: {'YES' if metrics.rapid_progression else 'NO'}
New Lesions Detected: {'YES' if metrics.new_lesions_detected else 'NO'}

"""

        # Add detailed flags
        if metrics.additional_metrics.get('clinical_flags'):
            flags = metrics.additional_metrics['clinical_flags']
            report += "Clinical Significance Flags:\n"
            for flag_name, flag_value in flags.items():
                if flag_value:
                    report += f"  - {flag_name.replace('_', ' ').title()}\n"

        # Add lesion tracking summary
        if metrics.additional_metrics.get('lesion_tracking'):
            tracking = metrics.additional_metrics['lesion_tracking']
            report += f"""
LESION TRACKING
---------------
Matched Lesions: {tracking['n_matched']}
New Lesions: {tracking['n_new']}
Disappeared Lesions: {tracking['n_disappeared']}
"""

        return report

    def export_for_knowledge_graph(self, metrics: TemporalMetrics) -> Dict:
        """
        Export temporal metrics in a format suitable for knowledge graph integration.

        Returns:
            Dictionary with nodes and edges for graph construction
        """
        nodes = []
        edges = []

        # Patient node
        patient_node = {
            'id': f'patient_{metrics.patient_id}',
            'type': 'Patient',
            'patient_id': metrics.patient_id,
        }
        nodes.append(patient_node)

        # Timepoint nodes
        t1_node = {
            'id': f'scan_{metrics.patient_id}_{metrics.timepoint_1.strftime("%Y%m%d")}',
            'type': 'MRI_Scan',
            'date': metrics.timepoint_1.isoformat(),
            'patient_id': metrics.patient_id,
        }
        t2_node = {
            'id': f'scan_{metrics.patient_id}_{metrics.timepoint_2.strftime("%Y%m%d")}',
            'type': 'MRI_Scan',
            'date': metrics.timepoint_2.isoformat(),
            'patient_id': metrics.patient_id,
        }
        nodes.extend([t1_node, t2_node])

        # Temporal analysis node
        temporal_node = {
            'id': f'temporal_analysis_{metrics.patient_id}_{metrics.timepoint_2.strftime("%Y%m%d")}',
            'type': 'TemporalAnalysis',
            'time_delta_days': metrics.time_delta_days,
            'total_volume_change': metrics.total_volume_change,
            'growth_rate': metrics.total_growth_rate,
            'treatment_response': metrics.treatment_response,
            'rapid_progression': metrics.rapid_progression,
            'new_lesions': metrics.new_lesions_detected,
        }
        nodes.append(temporal_node)

        # Edges
        edges.extend([
            {'source': patient_node['id'], 'target': t1_node['id'], 'type': 'HAS_SCAN'},
            {'source': patient_node['id'], 'target': t2_node['id'], 'type': 'HAS_SCAN'},
            {'source': t1_node['id'], 'target': t2_node['id'], 'type': 'FOLLOWED_BY',
             'time_delta_days': metrics.time_delta_days},
            {'source': t1_node['id'], 'target': temporal_node['id'], 'type': 'BASELINE_FOR'},
            {'source': t2_node['id'], 'target': temporal_node['id'], 'type': 'FOLLOWUP_FOR'},
        ])

        # Lesion tracking nodes and edges
        if metrics.additional_metrics.get('lesion_tracking'):
            tracking = metrics.additional_metrics['lesion_tracking']

            # Matched lesions
            for i, match in enumerate(tracking['matched']):
                lesion_node = {
                    'id': f'lesion_{metrics.patient_id}_tracked_{i}',
                    'type': 'TrackedLesion',
                    'volume_change': match['volume_change'],
                    'displacement': match['distance'],
                }
                nodes.append(lesion_node)
                edges.extend([
                    {'source': t1_node['id'], 'target': lesion_node['id'], 'type': 'HAS_LESION'},
                    {'source': t2_node['id'], 'target': lesion_node['id'], 'type': 'HAS_LESION'},
                ])

            # New lesions
            for i, lesion in enumerate(tracking['new']):
                lesion_node = {
                    'id': f'lesion_{metrics.patient_id}_new_{i}',
                    'type': 'NewLesion',
                    'volume': lesion['volume'],
                    'centroid': lesion['centroid'],
                }
                nodes.append(lesion_node)
                edges.append({'source': t2_node['id'], 'target': lesion_node['id'],
                              'type': 'HAS_NEW_LESION'})

        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'patient_id': metrics.patient_id,
                'analysis_type': 'temporal_quantitative',
                'timestamp': datetime.now().isoformat(),
            }
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_temporal_scan_from_nifti(scan_id: str, patient_id: str,
                                  acquisition_date: datetime,
                                  modality_paths: Dict[str, str],
                                  segmentation_path: Optional[str] = None) -> TemporalScan:
    """
    Load a temporal scan from NIfTI files.

    Args:
        scan_id: Unique scan identifier
        patient_id: Patient identifier
        acquisition_date: Scan acquisition date
        modality_paths: Dictionary mapping modality names to file paths
        segmentation_path: Optional path to segmentation file

    Returns:
        TemporalScan object
    """
    modalities = {}
    for modality_name, path in modality_paths.items():
        nii = nib.load(path)
        modalities[modality_name] = nii.get_fdata()

    segmentation = None
    if segmentation_path:
        seg_nii = nib.load(segmentation_path)
        segmentation = seg_nii.get_fdata()

    return TemporalScan(
        scan_id=scan_id,
        patient_id=patient_id,
        acquisition_date=acquisition_date,
        modalities=modalities,
        segmentation=segmentation,
    )


def example_usage():
    """Example demonstrating the usage of QuantitativeTemporalModule."""

    print("=" * 80)
    print("QUANTITATIVE TEMPORAL MODULE - EXAMPLE USAGE")
    print("=" * 80)

    # Simulate two timepoints with synthetic data
    print("\n1. Creating synthetic temporal scans...")

    # Baseline scan (T1)
    scan_t1 = TemporalScan(
        scan_id='scan_001_baseline',
        patient_id='patient_001',
        acquisition_date=datetime(2024, 1, 15),
        modalities={
            'T1': np.random.randn(128, 128, 64),
            'T2': np.random.randn(128, 128, 64),
            'FLAIR': np.random.randn(128, 128, 64),
        },
        segmentation=np.zeros((128, 128, 64)),
    )

    # Create a synthetic tumor in baseline
    center = np.array([64, 64, 32])
    for i in range(128):
        for j in range(128):
            for k in range(64):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2)
                if dist < 15:  # Tumor radius
                    scan_t1.segmentation[i, j, k] = 4  # Enhancing
                elif dist < 20:
                    scan_t1.segmentation[i, j, k] = 2  # Edema

    # Follow-up scan (T2) - 3 months later with growth
    scan_t2 = TemporalScan(
        scan_id='scan_001_followup',
        patient_id='patient_001',
        acquisition_date=datetime(2024, 4, 15),
        modalities={
            'T1': np.random.randn(128, 128, 64),
            'T2': np.random.randn(128, 128, 64),
            'FLAIR': np.random.randn(128, 128, 64),
        },
        segmentation=np.zeros((128, 128, 64)),
    )

    # Simulate tumor growth
    for i in range(128):
        for j in range(128):
            for k in range(64):
                dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2 + (k - center[2]) ** 2)
                if dist < 18:  # Grown tumor
                    scan_t2.segmentation[i, j, k] = 4
                elif dist < 25:
                    scan_t2.segmentation[i, j, k] = 2

    # Initialize module
    print("2. Initializing Quantitative Temporal Module...")
    temporal_module = QuantitativeTemporalModule(voxel_spacing=(1.0, 1.0, 1.0))

    # Perform analysis
    print("3. Analyzing temporal changes...")
    metrics = temporal_module.analyze_temporal_pair(scan_t1, scan_t2)

    # Generate report
    print("\n4. Generating clinical report...\n")
    report = temporal_module.generate_temporal_report(metrics)
    print(report)

    # Export for knowledge graph
    print("5. Exporting for knowledge graph integration...")
    kg_data = temporal_module.export_for_knowledge_graph(metrics)
    print(f"   Created {len(kg_data['nodes'])} nodes and {len(kg_data['edges'])} edges")

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
