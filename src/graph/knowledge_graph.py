# /graph/knowledge_graph.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


class ClinicalKnowledgeGraph:
    """
    Multimodal knowledge graph for clinical data.
    Integrates MRI imaging, segmentations, and text.
    Node types:
      - Patient
      - MRIVolume
      - Lesion
      - Modality
      - ClinicalNote
      - Observation
    """

    def __init__(self) -> None:
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.node_counter: int = 0

    # ----------------------------
    # Utilities
    # ----------------------------
    def _get_node_id(self, prefix: str) -> str:
        """Generate unique node ID like 'patient1', 'mri2', ..."""
        self.node_counter += 1
        return f"{prefix}{self.node_counter}"

    @staticmethod
    def _to_list_embedding(embedding: Any) -> Optional[List[float]]:
        """Convert numpy/torch/list embeddings to JSON-serializable list[float]."""
        if embedding is None:
            return None
        if isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32).reshape(-1).tolist()
        # torch tensor support without importing torch here
        if hasattr(embedding, "detach") and hasattr(embedding, "cpu") and hasattr(embedding, "numpy"):
            return embedding.detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()
        if isinstance(embedding, list):
            return [float(x) for x in embedding]
        return None

    # ----------------------------
    # Node creators
    # ----------------------------
    def add_patient(self, patient_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add patient node.

        Args:
            patient_id: Unique patient identifier
            metadata: Additional patient information
        """
        node_id = self._get_node_id("patient")
        self.graph.add_node(
            node_id,
            type="Patient",
            patient_id=str(patient_id),
            **(metadata or {}),
        )
        return node_id

    def add_modality(self, modality_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add modality node (T1/T2/FLAIR/T1CE etc.)
        Useful if you want modality as a shared node rather than only a string property.
        """
        node_id = self._get_node_id("mod")
        self.graph.add_node(
            node_id,
            type="Modality",
            name=str(modality_name),
            **(metadata or {}),
        )
        return node_id

    def add_mri_volume(
        self,
        patient_node: str,
        modality: str,
        embedding: Any,
        metadata: Optional[Dict[str, Any]] = None,
        modality_node: Optional[str] = None,
    ) -> str:
        """
        Add MRI volume node.

        Args:
            patient_node: Parent patient node ID
            modality: MRI modality (T1, T2, etc.)
            embedding: Volume embeddings vector
            metadata: Additional metadata
            modality_node: Optional Modality node to connect
        """
        node_id = self._get_node_id("mri")
        self.graph.add_node(
            node_id,
            type="MRIVolume",
            modality=str(modality),
            embedding=self._to_list_embedding(embedding),
            **(metadata or {}),
        )

        # patient -> mri
        self.graph.add_edge(
            patient_node,
            node_id,
            relationship="has_mri",
            modality=str(modality),
        )

        # modality -> mri (optional normalized schema)
        if modality_node is not None:
            self.graph.add_edge(
                modality_node,
                node_id,
                relationship="describes",
            )

        return node_id

    def add_lesion(
        self,
        mri_node: str,
        lesion_type: str,
        embedding: Any,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add lesion node from segmentation.

        Args:
            mri_node: Parent MRI volume node
            lesion_type: enhancing/necrotic/edema/etc.
            embedding: Lesion region embeddings
            properties: Morphological properties (volume, location, bbox, centroid, etc.)
        """
        node_id = self._get_node_id("lesion")
        self.graph.add_node(
            node_id,
            type="Lesion",
            lesion_type=str(lesion_type),
            embedding=self._to_list_embedding(embedding),
            **(properties or {}),
        )

        self.graph.add_edge(
            mri_node,
            node_id,
            relationship="contains_lesion",
            lesion_type=str(lesion_type),
        )
        return node_id

    def add_clinical_note(
        self,
        patient_node: str,
        note_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a clinical note node.
        """
        node_id = self._get_node_id("note")
        self.graph.add_node(
            node_id,
            type="ClinicalNote",
            text=str(note_text),
            **(metadata or {}),
        )
        self.graph.add_edge(
            patient_node,
            node_id,
            relationship="has_clinical_note",
        )
        return node_id

    def add_clinical_observation(
        self,
        patient_node: str,
        text: str,
        embedding: Any,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add clinical observation from text.
        """
        node_id = self._get_node_id("obs")
        self.graph.add_node(
            node_id,
            type="Observation",
            text=str(text),
            embedding=self._to_list_embedding(embedding),
            category=category,
            **(metadata or {}),
        )
        self.graph.add_edge(
            patient_node,
            node_id,
            relationship="has_observation",
        )
        return node_id

    # ----------------------------
    # Relationships
    # ----------------------------
    def add_spatial_relationship(self, lesion1_node: str, lesion2_node: str, relationship_type: str) -> None:
        """
        Add spatial relationship between lesions: adjacent/overlapping/contained_in/etc.
        """
        self.graph.add_edge(
            lesion1_node,
            lesion2_node,
            relationship="spatial",
            spatial_type=str(relationship_type),
        )

    def add_temporal_relationship(self, node1: str, node2: str, time_diff_days: float) -> None:
        """
        Add temporal relationship between nodes: earlier -> later with time difference in days.
        """
        self.graph.add_edge(
            node1,
            node2,
            relationship="temporal",
            time_difference_days=float(time_diff_days),
        )

    # ----------------------------
    # Query helpers
    # ----------------------------
    def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None, max_hops: int = 1) -> List[str]:
        """
        Get neighboring nodes within max_hops.
        If relationship_type is provided, filters edges by edge attribute 'relationship'.
        """
        if max_hops < 1:
            return []

        if max_hops == 1:
            neighbors = list(self.graph.successors(node_id))
            if relationship_type is None:
                return neighbors

            filtered: List[str] = []
            for n in neighbors:
                # MultiDiGraph can have multiple edges between same nodes
                edge_dict = self.graph.get_edge_data(node_id, n, default={})
                if any(ed.get("relationship") == relationship_type for ed in edge_dict.values()):
                    filtered.append(n)
            return filtered

        # multi-hop BFS
        all_neighbors = set()
        frontier = {node_id}

        for _ in range(max_hops):
            next_frontier = set()
            for n in frontier:
                for nbr in self.get_neighbors(n, relationship_type=relationship_type, max_hops=1):
                    if nbr not in all_neighbors:
                        next_frontier.add(nbr)
            all_neighbors.update(next_frontier)
            frontier = next_frontier

        all_neighbors.discard(node_id)
        return list(all_neighbors)

    # ----------------------------
    # Persistence
    # ----------------------------
    def save(self, filepath: str | Path) -> None:
        """Save graph to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)  # JSON-serializable
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """
        Load knowledge graph from JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        obj = cls()
        obj.graph = nx.node_link_graph(
            data,
            directed=True,
            multigraph=True
        )
        return obj


    def add_lesion_region(
        self,
        lesion_node: str,
        region_name: str,
        voxels: int,
        pct: float | None = None,
    ) -> str:
        """
        Add tumor subregion node (WT / TC / ET).

        region_name:
            - WT (whole tumor)
            - TC (tumor core)
            - ET (enhancing tumor)

        Creates:
            Lesion --has_region--> Region
        """

        node_id = self._get_node_id("region")

        self.graph.add_node(
            node_id,
            type="Region",
            region=str(region_name),
            voxels=int(voxels),
            pct=float(pct) if pct is not None else None,
        )

        self.graph.add_edge(
            lesion_node,
            node_id,
            relationship="has_region",
            region=str(region_name),
        )

        return node_id



if __name__ == "__main__":
    import numpy as np

    print("Running knowledge graph test...")

    kg = ClinicalKnowledgeGraph()

    p = kg.add_patient(
        "BraTS20_Training_001",
        metadata={"age": 58, "sex": "M"},
    )

    mod_t1 = kg.add_modality("T1")

    mri = kg.add_mri_volume(
        patient_node=p,
        modality="T1",
        embedding=np.random.randn(768).astype(np.float32),
        metadata={"scan_date": "2020-01-01"},
        modality_node=mod_t1,
    )

    lesion = kg.add_lesion(
        mri_node=mri,
        lesion_type="enhancing",
        embedding=np.random.randn(768).astype(np.float32),
        properties={
            "tumor_volume_voxels": 12000,
            "bbox": [10, 50, 20, 90, 5, 60],
        },
    )

    obs = kg.add_clinical_observation(
        patient_node=p,
        text="Enhancing lesion with edema in right frontal lobe.",
        embedding=np.random.randn(768).astype(np.float32),
        category="radiology_finding",
    )

    kg.save("artifacts/kg.json")

    print("✅ Knowledge graph saved to artifacts/kg.json")
