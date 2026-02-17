"""
Test Suite for Quantitative Temporal Module
============================================

Comprehensive unit and integration tests for the temporal analysis components.

Run with: python test_temporal_module.py
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import torch

# Import modules to test
from quantitative_temporal_module import (
    QuantitativeTemporalModule,
    TemporalScan,
    VolumetricAnalyzer,
    MorphologicalAnalyzer,
    LesionTracker,
    ProgressionClassifier,
    TemporalFeatureExtractor,
)
from temporal_integration_guide import (
    MedRAGXTemporalIntegration,
    TemporalQueryRouter,
)


class TestVolumetricAnalyzer:
    """Test volumetric analysis functionality."""

    @staticmethod
    def test_volume_computation():
        """Test basic volume computation."""
        print("\n[TEST] Volumetric Analysis - Volume Computation")

        analyzer = VolumetricAnalyzer(voxel_spacing=(1.0, 1.0, 1.0))

        # Create simple cubic segmentation
        seg = np.zeros((10, 10, 10))
        seg[2:8, 2:8, 2:8] = 4  # 6x6x6 cube = 216 voxels

        volume = analyzer.compute_volume(seg, label=4)
        expected_volume = 6 * 6 * 6 * 1.0  # 216 mm³

        assert abs(volume - expected_volume) < 1e-6, \
            f"Expected {expected_volume} mm³, got {volume} mm³"

        print(f"   ✓ Volume computation correct: {volume:.1f} mm³")
        return True

    @staticmethod
    def test_volume_change():
        """Test volume change calculation."""
        print("\n[TEST] Volumetric Analysis - Volume Change")

        analyzer = VolumetricAnalyzer(voxel_spacing=(1.0, 1.0, 1.0))

        # Baseline
        seg_t1 = np.zeros((10, 10, 10))
        seg_t1[3:7, 3:7, 3:7] = 4  # 4x4x4 = 64 voxels

        # Follow-up (grown)
        seg_t2 = np.zeros((10, 10, 10))
        seg_t2[2:8, 2:8, 2:8] = 4  # 6x6x6 = 216 voxels

        changes = analyzer.compute_volume_change(seg_t1, seg_t2)

        expected_change = 216 - 64  # 152 mm³
        expected_pct = (expected_change / 64) * 100  # 237.5%

        assert abs(changes['enhancing_absolute'] - expected_change) < 1e-6, \
            "Absolute change incorrect"
        assert abs(changes['enhancing_percentage'] - expected_pct) < 0.1, \
            "Percentage change incorrect"

        print(f"   ✓ Absolute change: {changes['enhancing_absolute']:.1f} mm³")
        print(f"   ✓ Percentage change: {changes['enhancing_percentage']:.1f}%")
        return True


class TestMorphologicalAnalyzer:
    """Test morphological analysis functionality."""

    @staticmethod
    def test_centroid_computation():
        """Test centroid computation."""
        print("\n[TEST] Morphological Analysis - Centroid Computation")

        analyzer = MorphologicalAnalyzer()

        # Create segmentation with known centroid
        seg = np.zeros((10, 10, 10))
        seg[4:6, 4:6, 4:6] = 1  # 2x2x2 cube centered at (4.5, 4.5, 4.5)

        centroid = analyzer.compute_centroid(seg, label=1)
        expected_centroid = np.array([4.5, 4.5, 4.5])

        assert np.allclose(centroid, expected_centroid, atol=0.1), \
            f"Expected {expected_centroid}, got {centroid}"

        print(f"   ✓ Centroid correct: {centroid}")
        return True

    @staticmethod
    def test_centroid_displacement():
        """Test centroid displacement calculation."""
        print("\n[TEST] Morphological Analysis - Centroid Displacement")

        analyzer = MorphologicalAnalyzer()

        # Baseline at (5, 5, 5)
        seg_t1 = np.zeros((20, 20, 20))
        seg_t1[4:6, 4:6, 4:6] = 4

        # Moved to (10, 5, 5) - displaced by 5 voxels in x
        seg_t2 = np.zeros((20, 20, 20))
        seg_t2[9:11, 4:6, 4:6] = 4

        displacement = analyzer.compute_centroid_displacement(
            seg_t1, seg_t2, voxel_spacing=(1.0, 1.0, 1.0), label=4
        )

        expected_displacement = 5.0  # mm

        assert abs(displacement - expected_displacement) < 0.5, \
            f"Expected ~{expected_displacement} mm, got {displacement} mm"

        print(f"   ✓ Displacement: {displacement:.2f} mm")
        return True


class TestLesionTracker:
    """Test lesion tracking functionality."""

    @staticmethod
    def test_lesion_detection():
        """Test individual lesion detection."""
        print("\n[TEST] Lesion Tracking - Detection")

        tracker = LesionTracker(min_volume_threshold=10.0)

        # Create segmentation with 3 separate lesions
        seg = np.zeros((50, 50, 50))

        # Lesion 1
        seg[10:15, 10:15, 10:15] = 1  # 5x5x5 = 125 voxels

        # Lesion 2
        seg[30:33, 30:33, 30:33] = 1  # 3x3x3 = 27 voxels

        # Lesion 3
        seg[40:42, 40:42, 40:42] = 1  # 2x2x2 = 8 voxels (below threshold)

        lesions = tracker.detect_lesions(seg, voxel_volume=1.0)

        assert len(lesions) == 2, f"Expected 2 lesions, found {len(lesions)}"

        print(f"   ✓ Detected {len(lesions)} lesions (correctly filtered by threshold)")
        return True

    @staticmethod
    def test_lesion_tracking():
        """Test lesion tracking between timepoints."""
        print("\n[TEST] Lesion Tracking - Temporal Matching")

        tracker = LesionTracker(min_volume_threshold=10.0)

        # Timepoint 1: 2 lesions
        seg_t1 = np.zeros((50, 50, 50))
        seg_t1[10:15, 10:15, 10:15] = 1  # Lesion A
        seg_t1[30:33, 30:33, 30:33] = 1  # Lesion B

        # Timepoint 2: Lesion A grown, Lesion B same, Lesion C new
        seg_t2 = np.zeros((50, 50, 50))
        seg_t2[9:16, 9:16, 9:16] = 1  # Lesion A (grown)
        seg_t2[30:33, 30:33, 30:33] = 1  # Lesion B (stable)
        seg_t2[40:43, 40:43, 40:43] = 1  # Lesion C (new)

        lesions_t1 = tracker.detect_lesions(seg_t1, voxel_volume=1.0)
        lesions_t2 = tracker.detect_lesions(seg_t2, voxel_volume=1.0)

        tracking = tracker.track_lesions(lesions_t1, lesions_t2, max_distance=10.0)

        assert tracking['n_matched'] == 2, f"Expected 2 matched, got {tracking['n_matched']}"
        assert tracking['n_new'] == 1, f"Expected 1 new, got {tracking['n_new']}"
        assert tracking['n_disappeared'] == 0, f"Expected 0 disappeared, got {tracking['n_disappeared']}"

        print(f"   ✓ Matched: {tracking['n_matched']}, New: {tracking['n_new']}")
        return True


class TestProgressionClassifier:
    """Test clinical progression classification."""

    @staticmethod
    def test_response_classification():
        """Test RANO-based response classification."""
        print("\n[TEST] Progression Classification - RANO Criteria")

        classifier = ProgressionClassifier()

        # Test progressive disease (>25% increase)
        response = classifier.classify_response(
            volume_change_pct=30.0,
            new_lesions=False
        )
        assert response == 'progressive', f"Expected 'progressive', got '{response}'"
        print("   ✓ Progressive disease classification correct")

        # Test partial response (<-50% decrease)
        response = classifier.classify_response(
            volume_change_pct=-60.0,
            new_lesions=False
        )
        assert response == 'partial_response', f"Expected 'partial_response', got '{response}'"
        print("   ✓ Partial response classification correct")

        # Test stable disease
        response = classifier.classify_response(
            volume_change_pct=5.0,
            new_lesions=False
        )
        assert response == 'stable', f"Expected 'stable', got '{response}'"
        print("   ✓ Stable disease classification correct")

        # Test new lesions → progressive
        response = classifier.classify_response(
            volume_change_pct=5.0,
            new_lesions=True
        )
        assert response == 'progressive', f"Expected 'progressive' (new lesions), got '{response}'"
        print("   ✓ New lesion detection → progressive")

        return True


class TestTemporalFeatureExtractor:
    """Test neural temporal feature extraction."""

    @staticmethod
    def test_forward_pass():
        """Test forward pass through network."""
        print("\n[TEST] Neural Feature Extractor - Forward Pass")

        model = TemporalFeatureExtractor(input_channels=4, feature_dim=256)
        model.eval()

        # Create dummy input (batch=2, channels=4, D=32, H=32, W=32)
        scan_t1 = torch.randn(2, 4, 32, 32, 32)
        scan_t2 = torch.randn(2, 4, 32, 32, 32)

        with torch.no_grad():
            outputs = model(scan_t1, scan_t2)

        # Check output dimensions
        assert outputs['features_t1'].shape == (2, 256), \
            f"Wrong feature shape: {outputs['features_t1'].shape}"
        assert outputs['temporal_features'].shape == (2, 128), \
            f"Wrong temporal feature shape: {outputs['temporal_features'].shape}"

        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Feature dimensions correct")

        # Check prediction ranges
        assert outputs['progression_risk'].min() >= 0 and outputs['progression_risk'].max() <= 1, \
            "Progression risk not in [0,1]"
        assert outputs['response_score'].min() >= -1 and outputs['response_score'].max() <= 1, \
            "Response score not in [-1,1]"

        print(f"   ✓ Prediction ranges valid")
        return True


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    @staticmethod
    def test_full_temporal_analysis():
        """Test complete temporal analysis workflow."""
        print("\n[TEST] End-to-End Pipeline - Full Temporal Analysis")

        # Initialize module
        temporal_module = QuantitativeTemporalModule(voxel_spacing=(1.0, 1.0, 1.0))

        # Create synthetic scans
        seg_t1 = np.zeros((64, 64, 32))
        seg_t1[20:30, 20:30, 10:20] = 4  # Enhancing
        seg_t1[18:32, 18:32, 8:22] = 2  # Surrounding edema

        seg_t2 = np.zeros((64, 64, 32))
        seg_t2[19:32, 19:32, 9:22] = 4  # Grown enhancing
        seg_t2[17:34, 17:34, 7:24] = 2  # Increased edema

        scan_t1 = TemporalScan(
            scan_id='test_t1',
            patient_id='test_patient',
            acquisition_date=datetime(2024, 1, 1),
            modalities={
                'T1': np.random.randn(64, 64, 32),
                'T2': np.random.randn(64, 64, 32),
            },
            segmentation=seg_t1
        )

        scan_t2 = TemporalScan(
            scan_id='test_t2',
            patient_id='test_patient',
            acquisition_date=datetime(2024, 4, 1),
            modalities={
                'T1': np.random.randn(64, 64, 32),
                'T2': np.random.randn(64, 64, 32),
            },
            segmentation=seg_t2
        )

        # Perform analysis
        metrics = temporal_module.analyze_temporal_pair(scan_t1, scan_t2)

        # Validate metrics
        assert metrics.patient_id == 'test_patient', "Patient ID mismatch"
        assert metrics.time_delta_days > 0, "Time delta should be positive"
        assert metrics.total_volume_change != 0, "Should detect volume change"

        print(f"   ✓ Analysis completed successfully")
        print(f"   ✓ Time delta: {metrics.time_delta_days:.1f} days")
        print(f"   ✓ Volume change: {metrics.total_volume_change:+.1f} mm³")
        print(f"   ✓ Treatment response: {metrics.treatment_response}")

        # Test report generation
        report = temporal_module.generate_temporal_report(metrics)
        assert len(report) > 100, "Report should be substantial"
        assert 'TEMPORAL ANALYSIS REPORT' in report, "Report header missing"

        print(f"   ✓ Clinical report generated ({len(report)} chars)")

        # Test knowledge graph export
        kg_data = temporal_module.export_for_knowledge_graph(metrics)
        assert len(kg_data['nodes']) > 0, "Should have nodes"
        assert len(kg_data['edges']) > 0, "Should have edges"

        print(f"   ✓ Knowledge graph exported ({len(kg_data['nodes'])} nodes)")

        return True

    @staticmethod
    def test_integration_layer():
        """Test MedRAG-X integration layer."""
        print("\n[TEST] Integration Layer - Patient Timeline Management")

        temporal_module = QuantitativeTemporalModule(voxel_spacing=(1.0, 1.0, 1.0))
        integration = MedRAGXTemporalIntegration(temporal_module)

        # Add multiple scans
        for i in range(3):
            scan = TemporalScan(
                scan_id=f'scan_{i}',
                patient_id='patient_001',
                acquisition_date=datetime(2024, 1, 1) + timedelta(days=i * 90),
                modalities={'T1': np.random.randn(32, 32, 16)},
                segmentation=np.random.randint(0, 5, (32, 32, 16))
            )
            integration.add_scan_to_timeline(scan)

        # Verify timeline
        timeline = integration.get_patient_timeline('patient_001')
        assert len(timeline) == 3, f"Expected 3 scans, got {len(timeline)}"

        # Check chronological order
        for i in range(len(timeline) - 1):
            assert timeline[i].acquisition_date < timeline[i + 1].acquisition_date, \
                "Timeline not in chronological order"

        print(f"   ✓ Timeline management correct ({len(timeline)} scans)")

        return True


class TestQueryRouting:
    """Test temporal query routing."""

    @staticmethod
    def test_temporal_query_detection():
        """Test detection of temporal queries."""
        print("\n[TEST] Query Routing - Temporal Detection")

        router = TemporalQueryRouter()

        # Temporal queries
        temporal_queries = [
            "Has the tumor progressed since last scan?",
            "Compare baseline to follow-up",
            "Show me the change over time",
            "Is the patient responding to treatment?",
        ]

        for query in temporal_queries:
            routing = router.detect_temporal_query(query)
            assert routing['requires_temporal_analysis'], \
                f"Failed to detect temporal query: {query}"

        print(f"   ✓ All temporal queries detected correctly")

        # Non-temporal queries
        non_temporal_queries = [
            "What is the current tumor volume?",
            "Where is the lesion located?",
            "Describe the MRI findings",
        ]

        for query in non_temporal_queries:
            routing = router.detect_temporal_query(query)
            # These might or might not be temporal, just check it doesn't crash
            pass

        print(f"   ✓ Non-temporal queries processed")

        return True


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all test suites."""
    print("=" * 80)
    print("QUANTITATIVE TEMPORAL MODULE - TEST SUITE")
    print("=" * 80)

    test_results = []

    # Volumetric tests
    print("\n" + "─" * 80)
    print("VOLUMETRIC ANALYSIS TESTS")
    print("─" * 80)
    try:
        test_results.append(("Volume Computation", TestVolumetricAnalyzer.test_volume_computation()))
        test_results.append(("Volume Change", TestVolumetricAnalyzer.test_volume_change()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("Volumetric", False))

    # Morphological tests
    print("\n" + "─" * 80)
    print("MORPHOLOGICAL ANALYSIS TESTS")
    print("─" * 80)
    try:
        test_results.append(("Centroid", TestMorphologicalAnalyzer.test_centroid_computation()))
        test_results.append(("Displacement", TestMorphologicalAnalyzer.test_centroid_displacement()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("Morphological", False))

    # Lesion tracking tests
    print("\n" + "─" * 80)
    print("LESION TRACKING TESTS")
    print("─" * 80)
    try:
        test_results.append(("Detection", TestLesionTracker.test_lesion_detection()))
        test_results.append(("Tracking", TestLesionTracker.test_lesion_tracking()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("Lesion Tracking", False))

    # Classification tests
    print("\n" + "─" * 80)
    print("PROGRESSION CLASSIFICATION TESTS")
    print("─" * 80)
    try:
        test_results.append(("RANO Criteria", TestProgressionClassifier.test_response_classification()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("Classification", False))

    # Neural network tests
    print("\n" + "─" * 80)
    print("NEURAL FEATURE EXTRACTOR TESTS")
    print("─" * 80)
    try:
        test_results.append(("Forward Pass", TestTemporalFeatureExtractor.test_forward_pass()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("Neural Network", False))

    # End-to-end tests
    print("\n" + "─" * 80)
    print("END-TO-END PIPELINE TESTS")
    print("─" * 80)
    try:
        test_results.append(("Full Analysis", TestEndToEndPipeline.test_full_temporal_analysis()))
        test_results.append(("Integration", TestEndToEndPipeline.test_integration_layer()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("End-to-End", False))

    # Query routing tests
    print("\n" + "─" * 80)
    print("QUERY ROUTING TESTS")
    print("─" * 80)
    try:
        test_results.append(("Temporal Detection", TestQueryRouting.test_temporal_query_detection()))
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
        test_results.append(("Query Routing", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "─" * 80)
    print(f"Results: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")
    print("=" * 80)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
