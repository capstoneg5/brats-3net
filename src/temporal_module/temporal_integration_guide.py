"""
Integration Guide: Quantitative Temporal Module with MedRAG-X
==============================================================

This guide demonstrates how to integrate the Quantitative Temporal Module
into your MedRAG-X pipeline for temporal analysis and knowledge graph construction.

Author: MedRAG-X Team
"""

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

# Import the temporal module
from quantitative_temporal_module import (
    QuantitativeTemporalModule,
    TemporalScan,
    TemporalMetrics,
)


class MedRAGXTemporalIntegration:
    """
    Integration layer connecting the Quantitative Temporal Module
    with the MedRAG-X pipeline.
    """

    def __init__(self, temporal_module: QuantitativeTemporalModule):
        """
        Initialize the integration layer.

        Args:
            temporal_module: Instance of QuantitativeTemporalModule
        """
        self.temporal_module = temporal_module
        self.patient_timeline = {}  # patient_id -> list of TemporalScan

    def add_scan_to_timeline(self, scan: TemporalScan):
        """
        Add a scan to the patient's timeline.

        Args:
            scan: TemporalScan object to add
        """
        patient_id = scan.patient_id
        if patient_id not in self.patient_timeline:
            self.patient_timeline[patient_id] = []

        self.patient_timeline[patient_id].append(scan)
        # Sort by acquisition date
        self.patient_timeline[patient_id].sort(key=lambda x: x.acquisition_date)

    def get_patient_timeline(self, patient_id: str) -> List[TemporalScan]:
        """Get all scans for a patient in chronological order."""
        return self.patient_timeline.get(patient_id, [])

    def analyze_patient_progression(self, patient_id: str) -> List[TemporalMetrics]:
        """
        Analyze all consecutive timepoint pairs for a patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of TemporalMetrics for each consecutive pair
        """
        scans = self.get_patient_timeline(patient_id)

        if len(scans) < 2:
            raise ValueError(f"Patient {patient_id} has fewer than 2 scans")

        metrics_list = []
        for i in range(len(scans) - 1):
            metrics = self.temporal_module.analyze_temporal_pair(
                scans[i], scans[i + 1]
            )
            metrics_list.append(metrics)

        return metrics_list

    def build_temporal_knowledge_graph(self, patient_id: str) -> Dict:
        """
        Build a complete temporal knowledge graph for a patient.

        This integrates with the Clinical Multimodal Knowledge Graph
        construction in your pipeline (Task 5).

        Returns:
            Complete graph structure with temporal nodes and edges
        """
        metrics_list = self.analyze_patient_progression(patient_id)

        all_nodes = []
        all_edges = []

        for metrics in metrics_list:
            kg_data = self.temporal_module.export_for_knowledge_graph(metrics)
            all_nodes.extend(kg_data['nodes'])
            all_edges.extend(kg_data['edges'])

        # Remove duplicate nodes (keep latest version)
        unique_nodes = {}
        for node in all_nodes:
            unique_nodes[node['id']] = node

        return {
            'nodes': list(unique_nodes.values()),
            'edges': all_edges,
            'patient_id': patient_id,
            'num_timepoints': len(self.get_patient_timeline(patient_id)),
            'analysis_date': datetime.now().isoformat(),
        }

    def generate_temporal_features_for_retrieval(self, patient_id: str) -> Dict:
        """
        Generate temporal features for semantic retrieval (Task 4).

        These features can be used to enhance embedding generation and
        improve retrieval relevance for temporal queries.

        Returns:
            Dictionary of temporal features suitable for embedding
        """
        metrics_list = self.analyze_patient_progression(patient_id)

        if not metrics_list:
            return {}

        # Aggregate temporal features across all timepoints
        features = {
            'patient_id': patient_id,
            'num_scans': len(self.get_patient_timeline(patient_id)),
            'total_monitoring_duration_days': sum(m.time_delta_days for m in metrics_list),

            # Volumetric trends
            'max_growth_rate': max(m.total_growth_rate for m in metrics_list),
            'mean_growth_rate': np.mean([m.total_growth_rate for m in metrics_list]),
            'cumulative_volume_change': sum(m.total_volume_change for m in metrics_list),

            # Clinical status
            'current_response': metrics_list[-1].treatment_response,
            'ever_progressed': any(m.treatment_response == 'progressive' for m in metrics_list),
            'stable_periods': sum(1 for m in metrics_list if m.treatment_response == 'stable'),

            # Lesion dynamics
            'new_lesions_total': sum(
                m.additional_metrics.get('lesion_tracking', {}).get('n_new', 0)
                for m in metrics_list
            ),
            'rapid_progression_events': sum(1 for m in metrics_list if m.rapid_progression),

            # Morphological trends
            'max_hausdorff_distance': max(m.hausdorff_distance for m in metrics_list),
            'total_centroid_displacement': sum(m.centroid_displacement for m in metrics_list),
        }

        return features

    def create_temporal_augmented_prompt(self, patient_id: str,
                                         clinical_query: str) -> str:
        """
        Create a temporally-augmented prompt for LLM reasoning (Task 7).

        This integrates temporal context into the prompt assembly process.

        Args:
            patient_id: Patient identifier
            clinical_query: User's clinical question

        Returns:
            Enhanced prompt with temporal context
        """
        metrics_list = self.analyze_patient_progression(patient_id)

        if not metrics_list:
            return clinical_query

        # Build temporal summary
        temporal_summary = f"""
PATIENT TEMPORAL HISTORY (Patient {patient_id})
{'=' * 60}

Number of scans: {len(self.get_patient_timeline(patient_id))}
Total monitoring duration: {sum(m.time_delta_days for m in metrics_list):.0f} days

PROGRESSION TIMELINE:
"""

        for i, metrics in enumerate(metrics_list, 1):
            temporal_summary += f"""
Interval {i}: {metrics.timepoint_1.strftime('%Y-%m-%d')} → {metrics.timepoint_2.strftime('%Y-%m-%d')} ({metrics.time_delta_days:.0f} days)
  - Volume change: {metrics.total_volume_change:+.1f} mm³ ({metrics.additional_metrics['volume_changes']['total_percentage']:+.1f}%)
  - Growth rate: {metrics.total_growth_rate:+.2f} mm³/day
  - Treatment response: {metrics.treatment_response}
  - New lesions: {'Yes' if metrics.new_lesions_detected else 'No'}
  - Rapid progression: {'Yes' if metrics.rapid_progression else 'No'}
"""

        # Current status
        latest_metrics = metrics_list[-1]
        current_status = f"""
CURRENT STATUS (as of {latest_metrics.timepoint_2.strftime('%Y-%m-%d')}):
  - Latest response classification: {latest_metrics.treatment_response}
  - Current trend: {'Progressing' if latest_metrics.total_growth_rate > 0 else 'Stable/Responding'}
"""

        # Combine with clinical query
        augmented_prompt = f"""
{temporal_summary}
{current_status}

CLINICAL QUERY:
{clinical_query}

Please provide an evidence-based response considering the patient's temporal progression pattern.
When discussing changes, explicitly reference the specific timepoints and quantitative metrics provided above.
"""

        return augmented_prompt

    def add_temporal_guardrails(self, llm_response: str,
                                patient_id: str) -> Dict:
        """
        Add temporal-specific safety checks (Task 9 - Clinical Guardrails).

        Validates that LLM responses align with temporal evidence.

        Returns:
            Validation result with safety flags
        """
        metrics_list = self.analyze_patient_progression(patient_id)

        if not metrics_list:
            return {'validated': True, 'warnings': [], 'temporal_context': None}

        latest_metrics = metrics_list[-1]
        warnings = []

        # Check for contradictions with temporal data
        response_lower = llm_response.lower()

        # 1. Progression status validation
        if latest_metrics.treatment_response == 'progressive':
            if 'stable' in response_lower and 'not stable' not in response_lower:
                warnings.append(
                    f"⚠️ Response mentions 'stable' but patient shows progressive disease "
                    f"with {latest_metrics.total_growth_rate:+.2f} mm³/day growth rate"
                )

        # 2. Growth rate validation
        if latest_metrics.rapid_progression:
            if 'slow' in response_lower or 'indolent' in response_lower:
                warnings.append(
                    f"⚠️ Response suggests slow progression but data shows rapid progression "
                    f"(growth rate: {latest_metrics.total_growth_rate:.2f} mm³/day)"
                )

        # 3. New lesion validation
        if latest_metrics.new_lesions_detected:
            if 'no new lesions' in response_lower:
                warnings.append(
                    "⚠️ Response states no new lesions, but temporal analysis detected new lesions"
                )

        # 4. Temporal trend validation
        recent_trend = 'increasing' if latest_metrics.total_growth_rate > 10 else 'stable'
        if recent_trend == 'increasing':
            if 'decreasing' in response_lower or 'shrinking' in response_lower:
                warnings.append(
                    f"⚠️ Response suggests decrease but tumor is increasing "
                    f"({latest_metrics.total_volume_change:+.1f} mm³)"
                )

        return {
            'validated': len(warnings) == 0,
            'warnings': warnings,
            'temporal_context': {
                'latest_response': latest_metrics.treatment_response,
                'growth_rate': latest_metrics.total_growth_rate,
                'rapid_progression': latest_metrics.rapid_progression,
                'new_lesions': latest_metrics.new_lesions_detected,
            }
        }


class TemporalQueryRouter:
    """
    Routes temporal queries to appropriate processing pathways.

    Integrates with Task 1 (Query Routing) in your pipeline.
    """

    @staticmethod
    def detect_temporal_query(query: str) -> Dict:
        """
        Detect if a query requires temporal analysis.

        Returns:
            Dictionary with routing information
        """
        query_lower = query.lower()

        temporal_keywords = [
            'progression', 'change', 'growth', 'over time', 'compared to',
            'previous', 'baseline', 'follow-up', 'trend', 'evolution',
            'response to treatment', 'stable', 'worse', 'better',
            'since last scan', 'interval change'
        ]

        is_temporal = any(keyword in query_lower for keyword in temporal_keywords)

        # Determine specific temporal query type
        query_type = None
        if 'progression' in query_lower or 'growth' in query_lower:
            query_type = 'progression_assessment'
        elif 'response' in query_lower or 'treatment' in query_lower:
            query_type = 'treatment_response'
        elif 'compare' in query_lower or 'change' in query_lower:
            query_type = 'comparative_analysis'
        elif 'trend' in query_lower or 'over time' in query_lower:
            query_type = 'longitudinal_trend'

        return {
            'requires_temporal_analysis': is_temporal,
            'query_type': query_type,
            'priority': 'high' if is_temporal else 'normal',
            'suggested_retrieval_strategy': 'temporal_graph_traversal' if is_temporal else 'standard'
        }


class TemporalRetrievalEnhancer:
    """
    Enhances retrieval with temporal information.

    Integrates with Task 6 (Hybrid Retrieval) in your pipeline.
    """

    def __init__(self, integration_layer: MedRAGXTemporalIntegration):
        self.integration = integration_layer

    def enhance_semantic_retrieval(self, query: str, patient_id: str,
                                   base_retrieved_contexts: List[Dict]) -> List[Dict]:
        """
        Enhance retrieved contexts with temporal information.

        Args:
            query: User query
            patient_id: Patient identifier
            base_retrieved_contexts: Contexts from base semantic retrieval

        Returns:
            Enhanced contexts with temporal features
        """
        # Get temporal features
        temporal_features = self.integration.generate_temporal_features_for_retrieval(patient_id)

        enhanced_contexts = []
        for context in base_retrieved_contexts:
            # Add temporal metadata
            enhanced_context = context.copy()
            enhanced_context['temporal_features'] = temporal_features

            # Add temporal relevance score
            temporal_relevance = self._compute_temporal_relevance(
                query, context, temporal_features
            )
            enhanced_context['temporal_relevance'] = temporal_relevance

            # Boost overall relevance if temporally relevant
            if 'relevance_score' in enhanced_context:
                enhanced_context['relevance_score'] *= (1 + 0.3 * temporal_relevance)

            enhanced_contexts.append(enhanced_context)

        # Re-sort by enhanced relevance
        enhanced_contexts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        return enhanced_contexts

    def _compute_temporal_relevance(self, query: str, context: Dict,
                                    temporal_features: Dict) -> float:
        """Compute how relevant temporal features are to the query."""
        query_lower = query.lower()
        relevance = 0.0

        # Check for temporal keywords in query
        if any(word in query_lower for word in ['change', 'progression', 'growth']):
            relevance += 0.3

        # Check if context mentions timepoints
        if context.get('has_temporal_reference', False):
            relevance += 0.2

        # Check for treatment response queries
        if 'response' in query_lower or 'treatment' in query_lower:
            if temporal_features.get('current_response'):
                relevance += 0.3

        # Check for stability queries
        if 'stable' in query_lower:
            if temporal_features.get('stable_periods', 0) > 0:
                relevance += 0.2

        return min(relevance, 1.0)

    def temporal_graph_traversal(self, patient_id: str,
                                 query_embedding: np.ndarray,
                                 max_hops: int = 3) -> List[Dict]:
        """
        Perform graph traversal prioritizing temporal relationships.

        This implements temporal-aware structural retrieval to complement
        your semantic retrieval in the hybrid strategy.

        Returns:
            List of temporally-connected contexts
        """
        # Build temporal graph
        temporal_kg = self.integration.build_temporal_knowledge_graph(patient_id)

        # Simulate graph traversal (in practice, use Neo4j or NetworkX)
        temporally_connected_contexts = []

        # Start from most recent scan
        scans = self.integration.get_patient_timeline(patient_id)
        if not scans:
            return []

        latest_scan = scans[-1]

        # Add contexts from latest scan and traverse backwards
        for scan in reversed(scans):
            context = {
                'scan_id': scan.scan_id,
                'acquisition_date': scan.acquisition_date.isoformat(),
                'modalities': list(scan.modalities.keys()),
                'temporal_distance': (latest_scan.acquisition_date - scan.acquisition_date).days,
            }
            temporally_connected_contexts.append(context)

        return temporally_connected_contexts


# ============================================================================
# EXAMPLE: END-TO-END INTEGRATION WITH MedRAG-X
# ============================================================================

def example_medragx_integration():
    """
    Complete example showing integration with MedRAG-X pipeline.
    """
    print("=" * 80)
    print("MedRAG-X TEMPORAL INTEGRATION - COMPLETE EXAMPLE")
    print("=" * 80)

    # 1. Initialize components
    print("\n1. Initializing temporal module and integration layer...")
    temporal_module = QuantitativeTemporalModule(voxel_spacing=(1.0, 1.0, 1.0))
    integration = MedRAGXTemporalIntegration(temporal_module)

    # 2. Simulate patient timeline with 3 scans
    print("2. Creating patient timeline with 3 scans...")
    patient_id = 'patient_001'

    # Create synthetic scans
    for i, months_offset in enumerate([0, 3, 6]):
        scan = TemporalScan(
            scan_id=f'scan_{i + 1}',
            patient_id=patient_id,
            acquisition_date=datetime(2024, 1, 1) + timedelta(days=months_offset * 30),
            modalities={
                'T1': np.random.randn(64, 64, 32),
                'T2': np.random.randn(64, 64, 32),
                'FLAIR': np.random.randn(64, 64, 32),
            },
            segmentation=np.random.randint(0, 5, (64, 64, 32)),
        )
        integration.add_scan_to_timeline(scan)

    # 3. Query routing
    print("\n3. Testing temporal query routing...")
    clinical_query = "Has the tumor progressed since the last scan?"

    router = TemporalQueryRouter()
    routing_info = router.detect_temporal_query(clinical_query)
    print(f"   Query type: {routing_info['query_type']}")
    print(f"   Requires temporal analysis: {routing_info['requires_temporal_analysis']}")

    # 4. Temporal analysis
    print("\n4. Performing temporal analysis...")
    try:
        metrics_list = integration.analyze_patient_progression(patient_id)
        print(f"   Analyzed {len(metrics_list)} temporal intervals")
    except Exception as e:
        print(f"   Note: {e}")
        print("   (This is expected with random synthetic data)")

    # 5. Knowledge graph construction
    print("\n5. Building temporal knowledge graph...")
    try:
        temporal_kg = integration.build_temporal_knowledge_graph(patient_id)
        print(f"   Created graph with {len(temporal_kg['nodes'])} nodes and {len(temporal_kg['edges'])} edges")
    except Exception as e:
        print(f"   Note: {e}")

    # 6. Temporal feature extraction
    print("\n6. Extracting temporal features for retrieval...")
    try:
        temporal_features = integration.generate_temporal_features_for_retrieval(patient_id)
        print(f"   Extracted {len(temporal_features)} temporal features")
        print(f"   Features: {list(temporal_features.keys())[:5]}...")
    except Exception as e:
        print(f"   Note: {e}")

    # 7. Prompt augmentation
    print("\n7. Creating temporally-augmented prompt...")
    try:
        augmented_prompt = integration.create_temporal_augmented_prompt(
            patient_id, clinical_query
        )
        print(f"   Generated prompt with {len(augmented_prompt)} characters")
        print(f"   Preview: {augmented_prompt[:200]}...")
    except Exception as e:
        print(f"   Note: {e}")

    # 8. Retrieval enhancement
    print("\n8. Enhancing retrieval with temporal context...")
    retrieval_enhancer = TemporalRetrievalEnhancer(integration)

    # Simulate base retrieved contexts
    base_contexts = [
        {'text': 'Previous MRI showed...', 'relevance_score': 0.8},
        {'text': 'Patient history indicates...', 'relevance_score': 0.6},
    ]

    try:
        enhanced_contexts = retrieval_enhancer.enhance_semantic_retrieval(
            clinical_query, patient_id, base_contexts
        )
        print(f"   Enhanced {len(enhanced_contexts)} contexts")
    except Exception as e:
        print(f"   Note: {e}")

    # 9. Guardrails validation
    print("\n9. Applying temporal guardrails...")
    sample_response = "The tumor appears stable with no significant changes."

    try:
        validation = integration.add_temporal_guardrails(sample_response, patient_id)
        print(f"   Validation passed: {validation['validated']}")
        if validation['warnings']:
            print(f"   Warnings: {len(validation['warnings'])}")
    except Exception as e:
        print(f"   Note: {e}")

    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLE COMPLETE")
    print("=" * 80)
    print("\nNext steps for your team:")
    print("  - Integrate with your segmentation module (Task 3)")
    print("  - Connect to embedding generation (Task 4)")
    print("  - Add to knowledge graph construction (Task 5)")
    print("  - Enhance hybrid retrieval (Task 6)")
    print("  - Augment prompt assembly (Task 7)")
    print("  - Add to LLM reasoning (Task 8)")
    print("  - Integrate with guardrails (Task 9)")


if __name__ == "__main__":
    example_medragx_integration()
