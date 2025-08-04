"""DNA sequence analyzer providing various sequence analysis functions"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis


@dataclass
class SequenceFeature:
    """Sequence functional element"""
    name: str
    start: int
    end: int
    sequence: str
    feature_type: str
    confidence: float = 0.0
    description: str = ""


@dataclass
class SequenceAnalysisResult:
    """Sequence analysis result"""
    sequence: str
    length: int
    gc_content: float
    molecular_weight: float
    features: List[SequenceFeature]
    issues: List[str]
    recommendations: List[str]
    quality_score: float


class SequenceAnalyzer:
    """DNA sequence analyzer"""
    
    def __init__(self):
        # Common biological functional element patterns
        self.patterns = {
            "T7_promoter": r"TAATACGACTCACTATAG",
            "RBS_strong": r"AGGAGG[ATCG]{1,3}",
            "RBS_weak": r"AAGGAG[ATCG]{1,3}",
            "start_codon": r"ATG",
            "stop_codon": r"(TAA|TAG|TGA)",
            "chi_site": r"GCTGGTGG",
            "poly_T": r"T{4,}",
            "poly_G": r"G{4,}",
            "hairpin_stem": r"([ATCG]{4,}).*?\1"
        }
    
    def analyze_sequence(self, sequence: str, context: str = "") -> SequenceAnalysisResult:
        """Complete sequence analysis
        
        Args:
            sequence: DNA sequence
            context: Analysis context
            
        Returns:
            Sequence analysis result
        """
        sequence = sequence.upper().strip()
        
        # Basic information calculation
        length = len(sequence)
        gc_content = self._calculate_gc_content(sequence)
        mol_weight = self._calculate_molecular_weight(sequence)
        
        # Functional element identification
        features = self._identify_features(sequence)
        
        # Problem detection
        issues = self._detect_issues(sequence, features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sequence, features, issues)
        
        # Quality scoring
        quality_score = self._calculate_quality_score(sequence, features, issues)
        
        return SequenceAnalysisResult(
            sequence=sequence,
            length=length,
            gc_content=gc_content,
            molecular_weight=mol_weight,
            features=features,
            issues=issues,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content"""
        try:
            return gc_fraction(sequence) * 100
        except:
            gc_count = sequence.count('G') + sequence.count('C')
            return (gc_count / len(sequence)) * 100 if sequence else 0.0
    
    def _calculate_molecular_weight(self, sequence: str) -> float:
        """Calculate molecular weight"""
        try:
            return molecular_weight(sequence, seq_type='DNA')
        except:
            # Simple estimation: approximately 330 Da per nucleotide
            return len(sequence) * 330.0
    
    def _identify_features(self, sequence: str) -> List[SequenceFeature]:
        """Identify functional elements"""
        features = []
        
        for feature_name, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, sequence, re.IGNORECASE))
            for match in matches:
                start, end = match.span()
                feature_seq = sequence[start:end]
                
                # Calculate confidence
                confidence = self._calculate_feature_confidence(feature_name, feature_seq)
                
                feature = SequenceFeature(
                    name=feature_name,
                    start=start,
                    end=end,
                    sequence=feature_seq,
                    feature_type=self._get_feature_type(feature_name),
                    confidence=confidence,
                    description=self._get_feature_description(feature_name)
                )
                features.append(feature)
        
        return sorted(features, key=lambda x: x.start)
    
    def _calculate_feature_confidence(self, feature_name: str, sequence: str) -> float:
        """Calculate confidence of functional elements"""
        confidence_map = {
            "T7_promoter": 0.95,
            "RBS_strong": 0.85,
            "RBS_weak": 0.70,
            "start_codon": 0.99,
            "stop_codon": 0.99,
            "chi_site": 0.90,
            "poly_T": 0.60,
            "poly_G": 0.60,
            "hairpin_stem": 0.50
        }
        return confidence_map.get(feature_name, 0.5)
    
    def _get_feature_type(self, feature_name: str) -> str:
        """Get functional element type"""
        type_map = {
            "T7_promoter": "promoter",
            "RBS_strong": "ribosome_binding_site",
            "RBS_weak": "ribosome_binding_site",
            "start_codon": "start_codon",
            "stop_codon": "stop_codon",
            "chi_site": "protection_element",
            "poly_T": "terminator_like",
            "poly_G": "structure_risk",
            "hairpin_stem": "secondary_structure"
        }
        return type_map.get(feature_name, "unknown")
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get functional element description"""
        descriptions = {
            "T7_promoter": "T7 RNA polymerase promoter for efficient transcription initiation",
            "RBS_strong": "Strong ribosome binding site promoting translation initiation",
            "RBS_weak": "Weak ribosome binding site with moderate translation efficiency",
            "start_codon": "Start codon ATG, translation initiation signal",
            "stop_codon": "Stop codon, translation termination signal",
            "chi_site": "Chi site protecting DNA from RecBCD degradation",
            "poly_T": "Poly-T sequence that may form transcription terminator",
            "poly_G": "Poly-G sequence that may form G-quadruplex structure",
            "hairpin_stem": "Potential hairpin structure that may affect transcription or translation"
        }
        return descriptions.get(feature_name, "Unknown functional element")
    
    def _detect_issues(self, sequence: str, features: List[SequenceFeature]) -> List[str]:
        """Detect sequence issues"""
        issues = []
        
        # Check sequence length
        if len(sequence) > 140:
            issues.append(f"Sequence length ({len(sequence)}bp) exceeds recommended 140bp limit")
        
        # Check GC content
        gc_content = self._calculate_gc_content(sequence)
        if gc_content < 30 or gc_content > 70:
            issues.append(f"GC content ({gc_content:.1f}%) is not within ideal range (30-70%)")
        
        # Check functional elements
        has_promoter = any(f.feature_type == "promoter" for f in features)
        has_rbs = any(f.feature_type == "ribosome_binding_site" for f in features)
        has_start = any(f.feature_type == "start_codon" for f in features)
        
        if not has_promoter:
            issues.append("Missing promoter element")
        if not has_rbs:
            issues.append("Missing ribosome binding site (RBS)")
        if not has_start:
            issues.append("Missing start codon ATG")
        
        # Check potential issues
        poly_g_features = [f for f in features if f.name == "poly_G"]
        if poly_g_features:
            issues.append("Poly-G sequences present, may form G-quadruplex structures")
        
        poly_t_features = [f for f in features if f.name == "poly_T"]
        if len(poly_t_features) > 1:
            issues.append("Multiple poly-T sequences present, may cause transcription termination")
        
        # Check distance between RBS and start codon
        rbs_features = [f for f in features if f.feature_type == "ribosome_binding_site"]
        start_features = [f for f in features if f.feature_type == "start_codon"]
        
        for rbs in rbs_features:
            for start in start_features:
                distance = start.start - rbs.end
                if distance < 3 or distance > 15:
                    issues.append(f"RBS-start codon distance ({distance}bp) not in optimal range (5-9bp)")
        
        return issues
    
    def _generate_recommendations(self, sequence: str, features: List[SequenceFeature], 
                                issues: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Generate recommendations based on issues
        for issue in issues:
            if "GC content" in issue:
                if "30" in issue and "below" in issue:
                    recommendations.append("Increase G and C nucleotide ratio to improve GC content")
                else:
                    recommendations.append("Reduce G and C nucleotide ratio to lower GC content")
            
            elif "Missing promoter" in issue:
                recommendations.append("Add T7 promoter sequence (TAATACGACTCACTATAGGG)")
            
            elif "Missing ribosome binding site" in issue:
                recommendations.append("Add strong RBS sequence (e.g., AGGAGG)")
            
            elif "Missing start codon" in issue:
                recommendations.append("Add ATG start codon at sequence end")
            
            elif "Poly-G" in issue:
                recommendations.append("Replace consecutive G sequences to avoid G-quadruplex formation")
            
            elif "Poly-T" in issue:
                recommendations.append("Reduce consecutive T sequences to avoid unexpected transcription termination")
            
            elif "distance" in issue and "RBS" in issue:
                recommendations.append("Adjust spacer length between RBS and start codon to 5-9bp")
        
        # General recommendations
        if not any(f.name == "chi_site" for f in features):
            recommendations.append("Consider adding Chi site (GCTGGTGG) to protect linear DNA")
        
        if len(sequence) < 50:
            recommendations.append("Sequence is short, consider adding 5'UTR to enhance translation efficiency")
        
        return recommendations
    
    def _calculate_quality_score(self, sequence: str, features: List[SequenceFeature], 
                               issues: List[str]) -> float:
        """Calculate sequence quality score (0-100)"""
        score = 100.0
        
        # Deduct points based on issues
        score -= len(issues) * 10
        
        # Add points based on functional elements
        essential_features = ["promoter", "ribosome_binding_site", "start_codon"]
        for feature_type in essential_features:
            if any(f.feature_type == feature_type for f in features):
                score += 5
        
        # GC content scoring
        gc_content = self._calculate_gc_content(sequence)
        if 40 <= gc_content <= 60:
            score += 10
        elif 30 <= gc_content <= 70:
            score += 5
        
        # Length scoring
        if 50 <= len(sequence) <= 120:
            score += 10
        elif len(sequence) <= 140:
            score += 5
        
        return max(0.0, min(100.0, score))