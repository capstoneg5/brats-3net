# Quantitative Temporal Module for MedRAG-X

## Overview

The **Quantitative Temporal Module** extends the MedRAG-X system with comprehensive temporal analysis capabilities for tracking brain MRI changes over time. This module computes quantitative progression metrics, tracks individual lesions, classifies treatment responses, and integrates temporal features into your multimodal knowledge graph.

## Features

### 1. **Volumetric Analysis**
- Compute tumor volumes for all BraTS segmentation components (enhancing, necrotic, edema)
- Calculate absolute and percentage volume changes
- Compute growth rates (mm³/day)

### 2. **Morphological Analysis**
- Track tumor centroid displacement
- Compute Hausdorff distance for shape changes
- Measure surface area changes
- Detect infiltrative growth patterns

### 3. **Intensity Analysis**
- Track mean intensity changes across modalities (T1, T2, FLAIR, T1ce)
- Measure intensity heterogeneity changes
- Quantify texture evolution

### 4. **Lesion Tracking**
- Detect and track individual lesions across timepoints
- Identify new lesions
- Track disappearing lesions
- Match lesions using centroid proximity

### 5. **Clinical Classification**
- RANO-based treatment response classification
- Rapid progression detection
- Clinical significance flags
- Uncertainty-aware assessment

### 6. **Neural Temporal Features**
- Deep learning-based temporal feature extraction
- Siamese network architecture for paired scan analysis
- Learned progression risk prediction

## Installation

### Prerequisites
```bash
pip install numpy scipy nibabel torch --break-system-packages
```

### Optional Dependencies (for full integration)
```bash
pip install networkx fastapi pydantic monai --break-system-packages
```

## Quick Start

### Basic Usage

```python
from quantitative_temporal_module import (
    QuantitativeTemporalModule,
    TemporalScan,
    load_temporal_scan_from_nifti
)
from datetime import datetime

# 1. Initialize the module
temporal_module = QuantitativeTemporalModule(
    voxel_spacing=(1.0, 1.0, 1.0)  # Adjust based on your data
)

# 2. Load scans
scan_baseline = load_temporal_scan_from_nifti(
    scan_id='scan_001_baseline',
    patient_id='patient_001',
    acquisition_date=datetime(2024, 1, 15),
    modality_paths={
        'T1': '/path/to/t1.nii.gz',
        'T2': '/path/to/t2.nii.gz',
        'FLAIR': '/path/to/flair.nii.gz',
        'T1ce': '/path/to/t1ce.nii.gz',
    },
    segmentation_path='/path/to/segmentation.nii.gz'
)

scan_followup = load_temporal_scan_from_nifti(
    scan_id='scan_001_followup',
    patient_id='patient_001',
    acquisition_date=datetime(2024, 4, 15),
    modality_paths={
        'T1': '/path/to/followup_t1.nii.gz',
        'T2': '/path/to/followup_t2.nii.gz',
        'FLAIR': '/path/to/followup_flair.nii.gz',
        'T1ce': '/path/to/followup_t1ce.nii.gz',
    },
    segmentation_path='/path/to/followup_segmentation.nii.gz'
)

# 3. Analyze temporal changes
metrics = temporal_module.analyze_temporal_pair(scan_baseline, scan_followup)

# 4. Generate clinical report
report = temporal_module.generate_temporal_report(metrics)
print(report)

# 5. Export for knowledge graph
kg_data = temporal_module.export_for_knowledge_graph(metrics)
```

## Integration with MedRAG-X Pipeline

### Task-by-Task Integration Guide

#### Task 1: Data Ingestion (Harshvardhan)
```python
# When ingesting BraTS data, create TemporalScan objects
from quantitative_temporal_module import TemporalScan

def ingest_brats_scan(nifti_paths, patient_metadata):
    """Ingest BraTS scan into temporal format."""
    scan = load_temporal_scan_from_nifti(
        scan_id=patient_metadata['scan_id'],
        patient_id=patient_metadata['patient_id'],
        acquisition_date=patient_metadata['date'],
        modality_paths=nifti_paths,
        segmentation_path=patient_metadata['seg_path']
    )
    return scan
```

#### Task 2-3: Preprocessing & Segmentation (Sameeka)
```python
# After segmentation, attach to TemporalScan
def add_segmentation_to_scan(scan, segmentation_output):
    """Add 3D UNet segmentation to temporal scan."""
    scan.segmentation = segmentation_output
    return scan
```

#### Task 4: Embedding Generation (Esaikiappan)
```python
from temporal_integration_guide import MedRAGXTemporalIntegration

# Generate temporal features for embedding
integration = MedRAGXTemporalIntegration(temporal_module)
temporal_features = integration.generate_temporal_features_for_retrieval(patient_id)

# Concatenate with image embeddings
combined_embedding = concatenate([image_embedding, temporal_features])
```

#### Task 5: Graph Construction (Vineeth)
```python
# Build temporal knowledge graph
temporal_kg = integration.build_temporal_knowledge_graph(patient_id)

# Merge with clinical knowledge graph
clinical_kg['nodes'].extend(temporal_kg['nodes'])
clinical_kg['edges'].extend(temporal_kg['edges'])
```

#### Task 6: Hybrid Retrieval (Vineeth)
```python
from temporal_integration_guide import TemporalRetrievalEnhancer

retrieval_enhancer = TemporalRetrievalEnhancer(integration)

# Enhance retrieved contexts
enhanced_contexts = retrieval_enhancer.enhance_semantic_retrieval(
    query=user_query,
    patient_id=patient_id,
    base_retrieved_contexts=semantic_results
)
```

#### Task 7: Prompt Assembly (Esaikiappan)
```python
# Create temporally-augmented prompt
augmented_prompt = integration.create_temporal_augmented_prompt(
    patient_id=patient_id,
    clinical_query=user_query
)

# Pass to LLM
llm_response = llm.generate(augmented_prompt)
```

#### Task 8: LLM Reasoning (Esaikiappan)
```python
# System prompt should include temporal awareness
system_prompt = """
You are a clinical AI assistant with access to longitudinal MRI data.
When discussing temporal changes, always:
1. Reference specific timepoints
2. Quote quantitative metrics
3. Classify using RANO criteria
4. Express uncertainty when appropriate
"""
```

#### Task 9: Guardrails (Harshvardhan)
```python
# Apply temporal guardrails
validation = integration.add_temporal_guardrails(
    llm_response=generated_response,
    patient_id=patient_id
)

if not validation['validated']:
    # Flag for review or regenerate
    print("Warnings:", validation['warnings'])
```

## Architecture Components

### Core Classes

1. **TemporalScan**
   - Container for MRI data at a single timepoint
   - Stores modalities, segmentation, metadata

2. **TemporalMetrics**
   - Comprehensive metrics from temporal analysis
   - Volume changes, growth rates, morphological changes
   - Clinical classification and significance flags

3. **VolumetricAnalyzer**
   - Computes volumes for tumor components
   - Calculates volume changes and growth rates

4. **MorphologicalAnalyzer**
   - Tracks shape changes via Hausdorff distance
   - Computes centroid displacement
   - Estimates surface area changes

5. **IntensityAnalyzer**
   - Analyzes intensity patterns across modalities
   - Measures heterogeneity changes

6. **LesionTracker**
   - Detects individual lesions
   - Tracks lesions across timepoints
   - Identifies new and disappeared lesions

7. **ProgressionClassifier**
   - RANO-based response classification
   - Clinical significance assessment

8. **TemporalFeatureExtractor** (Neural)
   - Deep learning model for temporal features
   - Siamese architecture for paired scans
   - Progression risk prediction

## Clinical Metrics Reference

### RANO Criteria Implementation

**Progressive Disease:**
- ≥25% increase in tumor volume, OR
- Any new lesions

**Partial Response:**
- ≥50% decrease in tumor volume
- No new lesions

**Stable Disease:**
- Changes not meeting above criteria

### Growth Rate Thresholds

- **Rapid Progression**: >100 mm³/day
- **Moderate Growth**: 10-100 mm³/day  
- **Slow Growth**: 0-10 mm³/day
- **Stable/Regressing**: <0 mm³/day

## Knowledge Graph Schema

### Node Types
- `Patient`: Patient entity
- `MRI_Scan`: Individual scan timepoint
- `TemporalAnalysis`: Analysis between timepoints
- `TrackedLesion`: Lesion tracked across time
- `NewLesion`: Newly detected lesion

### Edge Types
- `HAS_SCAN`: Patient → Scan
- `FOLLOWED_BY`: Scan → Scan (temporal sequence)
- `BASELINE_FOR`: Scan → Analysis
- `FOLLOWUP_FOR`: Scan → Analysis
- `HAS_LESION`: Scan → Lesion
- `HAS_NEW_LESION`: Scan → NewLesion

## Example Outputs

### Clinical Report
```
TEMPORAL ANALYSIS REPORT
========================
Patient ID: patient_001
Baseline Scan: 2024-01-15
Follow-up Scan: 2024-04-15
Time Interval: 90.0 days (3.0 months)

VOLUMETRIC CHANGES
------------------
Total Tumor Volume Change: +1250.5 mm³
Enhancing Component: +890.2 mm³
Necrotic Component: +245.8 mm³
Edema: +114.5 mm³

GROWTH DYNAMICS
---------------
Total Growth Rate: +13.89 mm³/day
Enhancing Growth Rate: +9.89 mm³/day

CLINICAL ASSESSMENT
-------------------
Treatment Response: PROGRESSIVE
Rapid Progression: NO
New Lesions Detected: YES
```

### Knowledge Graph Export
```json
{
  "nodes": [
    {
      "id": "patient_001",
      "type": "Patient",
      "patient_id": "patient_001"
    },
    {
      "id": "scan_001_20240115",
      "type": "MRI_Scan",
      "date": "2024-01-15",
      "patient_id": "patient_001"
    },
    {
      "id": "temporal_analysis_001",
      "type": "TemporalAnalysis",
      "time_delta_days": 90.0,
      "growth_rate": 13.89,
      "treatment_response": "progressive"
    }
  ],
  "edges": [
    {
      "source": "patient_001",
      "target": "scan_001_20240115",
      "type": "HAS_SCAN"
    }
  ]
}
```

## Performance Considerations

### Computational Complexity
- **Volumetric Analysis**: O(N) where N = number of voxels
- **Hausdorff Distance**: O(M*K) where M,K = number of surface points
- **Lesion Tracking**: O(L₁*L₂) where L = number of lesions

### Optimization Tips
1. **Downsample for Hausdorff**: Limit to 1000 surface points
2. **Parallel Processing**: Use multiprocessing for multiple patients
3. **Caching**: Cache computed volumes and features
4. **GPU Acceleration**: Use CUDA for neural feature extraction

## Testing

Run the built-in examples:

```bash
# Test core temporal module
python quantitative_temporal_module.py

# Test integration with MedRAG-X
python temporal_integration_guide.py
```

## Evaluation Metrics (for Capstone)

### For Your Project Report

1. **Segmentation Quality** (prerequisite)
   - Dice coefficient on BraTS validation set
   - IoU metrics

2. **Temporal Consistency**
   - Test-retest reliability on repeated scans
   - Inter-rater agreement with radiologist assessments

3. **Clinical Validity**
   - Agreement with radiologist RANO classifications
   - Correlation with clinical outcomes

4. **Retrieval Enhancement**
   - Compare Recall@K with/without temporal features
   - MRR improvement for temporal queries

5. **Knowledge Graph Utility**
   - Graph completeness (% of temporal relationships captured)
   - Query answering accuracy on temporal questions

## Troubleshooting

### Common Issues

**Issue**: Negative volumes or growth rates
- **Cause**: Segmentation quality issues or registration errors
- **Solution**: Verify segmentation quality, check for label consistency

**Issue**: High Hausdorff distances
- **Cause**: Segmentation artifacts or true rapid morphological change
- **Solution**: Visual inspection, consider smoothing segmentations

**Issue**: Memory errors with large volumes
- **Cause**: Loading full 3D volumes for multiple timepoints
- **Solution**: Process regions of interest, use chunking

## Team Integration Plan

### Week-by-Week Integration

**Week 1-2**: Core module testing
- Each team member tests with sample data
- Validate metrics against manual calculations

**Week 3-4**: Pipeline integration
- Vineeth: Knowledge graph integration
- Esaikiappan: Embedding and prompt integration
- Sameeka: Segmentation pipeline connection
- Harshvardhan: Guardrails integration

**Week 5-6**: End-to-end testing
- Test complete pipeline with BraTS dataset
- Evaluate temporal query performance

**Week 7-8**: Optimization and documentation
- Performance tuning
- Final evaluation and report

## Citation

If you use this module in your research or project, please reference:

```
MedRAG-X: Extending RAG-Anything for MRI-Aware Clinical Reasoning
Team C6-Team 5
Quantitative Temporal Module
2025
```

## Support & Contribution

For questions or issues:
1. Check the example usage in `temporal_integration_guide.py`
2. Review the inline documentation
3. Contact team members via ClickUp tickets

## License

This module is part of the MedRAG-X capstone project.

---

**Author**: MedRAG-X Team C6-Team 5  
**Date**: February 2025  
**Version**: 1.0
