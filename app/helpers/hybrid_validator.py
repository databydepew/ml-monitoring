"""
Hybrid Model Validation System

This module implements a hybrid approach to model validation that combines:
1. Real-time user feedback
2. Credit API data
3. Database records
4. Manual expert validation

Each source has a confidence score, and the final validation uses a weighted approach.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSource(Enum):
    USER_FEEDBACK = "user_feedback"
    CREDIT_API = "credit_api"
    DATABASE = "database"
    EXPERT_REVIEW = "expert_review"

@dataclass
class ValidationResult:
    prediction_id: str
    outcome: int
    confidence_score: float
    source: ValidationSource
    timestamp: datetime
    metadata: Dict

class ConfidenceScorer:
    """Assigns confidence scores to different validation sources"""
    
    def __init__(self):
        # Base confidence scores for each source
        self.base_scores = {
            ValidationSource.USER_FEEDBACK: 0.6,
            ValidationSource.CREDIT_API: 0.9,
            ValidationSource.DATABASE: 0.8,
            ValidationSource.EXPERT_REVIEW: 1.0
        }
        
        # Decay factors for aging data
        self.decay_rates = {
            ValidationSource.USER_FEEDBACK: 0.1,  # Fast decay
            ValidationSource.CREDIT_API: 0.05,    # Slow decay
            ValidationSource.DATABASE: 0.05,      # Slow decay
            ValidationSource.EXPERT_REVIEW: 0.02  # Very slow decay
        }
    
    def calculate_confidence(self, source: ValidationSource, age_days: int,
                           metadata: Dict) -> float:
        """Calculate confidence score based on source, age, and metadata"""
        base_score = self.base_scores[source]
        decay_rate = self.decay_rates[source]
        
        # Apply time decay
        time_factor = 1.0 / (1.0 + decay_rate * age_days)
        
        # Apply source-specific adjustments
        if source == ValidationSource.USER_FEEDBACK:
            # Higher confidence for experienced users
            user_experience = metadata.get('user_experience_years', 0)
            experience_factor = min(1.0, 0.5 + user_experience * 0.1)
            base_score *= experience_factor
            
        elif source == ValidationSource.CREDIT_API:
            # Lower confidence for partial data
            data_completeness = metadata.get('data_completeness', 1.0)
            base_score *= data_completeness
            
        elif source == ValidationSource.DATABASE:
            # Adjust based on data freshness
            days_since_update = metadata.get('days_since_update', 0)
            freshness_factor = 1.0 / (1.0 + 0.1 * days_since_update)
            base_score *= freshness_factor
            
        return min(1.0, base_score * time_factor)

class HybridValidator:
    """Combines multiple validation sources with confidence scoring"""
    
    def __init__(self):
        self.confidence_scorer = ConfidenceScorer()
        self.validation_results: Dict[str, List[ValidationResult]] = {}
        
        # Prometheus metrics
        self.validation_count = Counter(
            'hybrid_validation_count_total',
            'Number of validations by source',
            ['source']
        )
        self.confidence_scores = Histogram(
            'validation_confidence_scores',
            'Distribution of confidence scores',
            ['source']
        )
        self.weighted_accuracy = Gauge(
            'model_weighted_accuracy',
            'Model accuracy weighted by confidence scores'
        )
    
    async def add_validation(self, validation: ValidationResult):
        """Add a new validation result"""
        if validation.prediction_id not in self.validation_results:
            self.validation_results[validation.prediction_id] = []
            
        self.validation_results[validation.prediction_id].append(validation)
        
        # Update metrics
        self.validation_count.labels(validation.source.value).inc()
        self.confidence_scores.labels(validation.source.value).observe(
            validation.confidence_score
        )
        
        # Recalculate weighted accuracy
        await self.update_weighted_accuracy()
    
    def get_weighted_outcome(self, prediction_id: str) -> Optional[int]:
        """Get the weighted consensus outcome for a prediction"""
        if prediction_id not in self.validation_results:
            return None
            
        validations = self.validation_results[prediction_id]
        if not validations:
            return None
            
        # Calculate weighted votes
        weighted_votes = {0: 0.0, 1: 0.0}
        total_weight = 0.0
        
        for validation in validations:
            weight = validation.confidence_score
            weighted_votes[validation.outcome] += weight
            total_weight += weight
            
        if total_weight == 0:
            return None
            
        # Return the outcome with highest weighted votes
        return 1 if weighted_votes[1] > weighted_votes[0] else 0
    
    async def update_weighted_accuracy(self):
        """Update the weighted accuracy metric"""
        total_weighted_correct = 0.0
        total_weight = 0.0
        
        for prediction_id, validations in self.validation_results.items():
            # Get the predicted outcome (assuming stored in validation metadata)
            predicted_outcome = validations[0].metadata.get('predicted_outcome')
            if predicted_outcome is None:
                continue
                
            # Get weighted actual outcome
            actual_outcome = self.get_weighted_outcome(prediction_id)
            if actual_outcome is None:
                continue
                
            # Calculate weighted accuracy
            for validation in validations:
                weight = validation.confidence_score
                if predicted_outcome == actual_outcome:
                    total_weighted_correct += weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_accuracy = total_weighted_correct / total_weight
            self.weighted_accuracy.set(weighted_accuracy)

class ValidationCollector:
    """Collects validations from multiple sources"""
    
    def __init__(self, validator: HybridValidator):
        self.validator = validator
        self.db_engine = create_engine('postgresql://user:pass@localhost/loans')
        self.Session = sessionmaker(bind=self.db_engine)
    
    async def collect_user_feedback(self, feedback_data: Dict):
        """Process user feedback"""
        validation = ValidationResult(
            prediction_id=feedback_data['prediction_id'],
            outcome=feedback_data['actual_outcome'],
            confidence_score=self.validator.confidence_scorer.calculate_confidence(
                ValidationSource.USER_FEEDBACK,
                age_days=0,
                metadata={
                    'user_experience_years': feedback_data.get('user_experience', 0),
                    'predicted_outcome': feedback_data.get('predicted_outcome')
                }
            ),
            source=ValidationSource.USER_FEEDBACK,
            timestamp=datetime.now(),
            metadata=feedback_data
        )
        await self.validator.add_validation(validation)
    
    async def collect_credit_api_data(self, credit_data: Dict):
        """Process credit API data"""
        validation = ValidationResult(
            prediction_id=credit_data['loan_id'],
            outcome=1 if credit_data['status'] == 'GOOD_STANDING' else 0,
            confidence_score=self.validator.confidence_scorer.calculate_confidence(
                ValidationSource.CREDIT_API,
                age_days=0,
                metadata={
                    'data_completeness': credit_data.get('completeness', 1.0),
                    'predicted_outcome': credit_data.get('predicted_outcome')
                }
            ),
            source=ValidationSource.CREDIT_API,
            timestamp=datetime.now(),
            metadata=credit_data
        )
        await self.validator.add_validation(validation)
    
    async def collect_database_records(self):
        """Collect validation data from database"""
        session = self.Session()
        try:
            # Example query - adjust according to your schema
            records = session.execute("""
                SELECT 
                    prediction_id,
                    loan_status,
                    prediction_timestamp,
                    predicted_outcome
                FROM loan_records
                WHERE validation_status = 'CONFIRMED'
                AND updated_at >= NOW() - INTERVAL '1 day'
            """)
            
            for record in records:
                validation = ValidationResult(
                    prediction_id=record.prediction_id,
                    outcome=1 if record.loan_status == 'PAID' else 0,
                    confidence_score=self.validator.confidence_scorer.calculate_confidence(
                        ValidationSource.DATABASE,
                        age_days=0,
                        metadata={
                            'days_since_update': 0,
                            'predicted_outcome': record.predicted_outcome
                        }
                    ),
                    source=ValidationSource.DATABASE,
                    timestamp=datetime.now(),
                    metadata=dict(record)
                )
                await self.validator.add_validation(validation)
                
        finally:
            session.close()
    
    async def collect_expert_review(self, review_data: Dict):
        """Process expert review data"""
        validation = ValidationResult(
            prediction_id=review_data['prediction_id'],
            outcome=review_data['validated_outcome'],
            confidence_score=self.validator.confidence_scorer.calculate_confidence(
                ValidationSource.EXPERT_REVIEW,
                age_days=0,
                metadata={
                    'reviewer_experience_years': review_data.get('reviewer_experience', 0),
                    'predicted_outcome': review_data.get('predicted_outcome')
                }
            ),
            source=ValidationSource.EXPERT_REVIEW,
            timestamp=datetime.now(),
            metadata=review_data
        )
        await self.validator.add_validation(validation)

async def setup_hybrid_validation():
    """Setup and start the hybrid validation system"""
    validator = HybridValidator()
    collector = ValidationCollector(validator)
    
    # Example usage
    await collector.collect_user_feedback({
        'prediction_id': '123',
        'actual_outcome': 1,
        'user_experience': 5,
        'predicted_outcome': 1
    })
    
    await collector.collect_credit_api_data({
        'loan_id': '123',
        'status': 'GOOD_STANDING',
        'completeness': 0.95,
        'predicted_outcome': 1
    })
    
    await collector.collect_expert_review({
        'prediction_id': '123',
        'validated_outcome': 1,
        'reviewer_experience': 10,
        'predicted_outcome': 1
    })
    
    logger.info("Hybrid validation system initialized successfully")
    return validator, collector
