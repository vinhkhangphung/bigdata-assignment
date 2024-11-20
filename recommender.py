import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse.linalg import svds
from typing import List, Dict, Tuple


class LearningPathRecommender:
    def __init__(self, n_factors: int = 20):
        self.n_factors = n_factors
        self.problem_encoders = {}
        self.problem_features = None
        self.U = None
        self.sigma = None
        self.Vt = None
        self.problem_ids = None

    def _preprocess_problem_data(self, problems_df: pd.DataFrame) -> np.ndarray:
        """Preprocess problem features for matrix factorization."""
        # Handle content source
        self.problem_encoders['content_source'] = LabelEncoder()
        content_source_encoded = problems_df['content_source'].apply(
            lambda x: ','.join(eval(x))
        ).pipe(self.problem_encoders['content_source'].fit_transform)

        # Handle skills
        self.problem_encoders['skills'] = LabelEncoder()
        skills_encoded = problems_df['skills'].apply(
            lambda x: ','.join(eval(x))
        ).pipe(self.problem_encoders['skills'].fit_transform)

        # Handle problem type
        self.problem_encoders['problem_type'] = LabelEncoder()
        problem_type_encoded = problems_df['problem_type'].pipe(
            self.problem_encoders['problem_type'].fit_transform
        )

        # Handle tutoring types
        self.problem_encoders['tutoring_types'] = LabelEncoder()
        tutoring_types_encoded = problems_df['tutoring_types'].apply(
            lambda x: ','.join(eval(x))
        ).pipe(self.problem_encoders['tutoring_types'].fit_transform)

        # Combine numerical features
        numerical_features = problems_df[['mean_correct', 'mean_time_on_task']].fillna(0)
        scaler = StandardScaler()
        numerical_features_scaled = scaler.fit_transform(numerical_features)

        # Combine all features
        features = np.column_stack([
            content_source_encoded,
            skills_encoded,
            problem_type_encoded,
            tutoring_types_encoded,
            numerical_features_scaled
        ])

        return features

    def fit(self, problems_df: pd.DataFrame, student_problem_matrix: np.ndarray):
        """
        Fit the recommendation model using problem features and student-problem interaction matrix.

        Args:
            problems_df: DataFrame containing problem features
            student_problem_matrix: Matrix of student performances on problems (students x problems)
        """
        # Preprocess problem features
        self.problem_features = self._preprocess_problem_data(problems_df)
        self.problem_ids = problems_df['problem_id'].values

        # Normalize student-problem matrix
        student_problem_mean = np.mean(student_problem_matrix, axis=1)
        student_problem_demeaned = student_problem_matrix - student_problem_mean.reshape(-1, 1)

        # Perform SVD
        self.U, self.sigma, self.Vt = svds(student_problem_demeaned, k=self.n_factors)

        # Convert sigma to diagonal matrix
        self.sigma = np.diag(self.sigma)

    def recommend_next_problems(
            self,
            student_id: int,
            student_problem_matrix: np.ndarray,
            n_recommendations: int = 5
    ) -> List[Dict[str, float]]:
        """
        Recommend next problems for a student based on their performance history.

        Args:
            student_id: ID of the student
            student_problem_matrix: Matrix of student performances
            n_recommendations: Number of problems to recommend

        Returns:
            List of recommended problem IDs with predicted scores
        """
        # Reconstruct the utility matrix
        predicted_ratings = np.dot(
            np.dot(self.U, self.sigma),
            self.Vt
        )

        # Add back the mean ratings for each student
        student_mean = np.mean(student_problem_matrix[student_id])
        predicted_ratings[student_id] += student_mean

        # Get problems the student hasn't attempted
        unattempted_mask = student_problem_matrix[student_id] == 0
        unattempted_problems = predicted_ratings[student_id] * unattempted_mask

        # Get top N recommendations
        top_problem_indices = np.argsort(unattempted_problems)[::-1][:n_recommendations]

        recommendations = []
        for idx in top_problem_indices:
            recommendations.append({
                'problem_id': self.problem_ids[idx],
                'predicted_score': predicted_ratings[student_id][idx]
            })

        return recommendations

    def get_similar_problems(self, problem_id: float, n_similar: int = 5) -> List[Tuple[float, float]]:
        """
        Find similar problems based on problem features and student interaction patterns.

        Args:
            problem_id: ID of the reference problem
            n_similar: Number of similar problems to return

        Returns:
            List of tuples containing (problem_id, similarity_score)
        """
        problem_idx = np.where(self.problem_ids == problem_id)[0][0]

        # Combine latent factors and problem features
        problem_vectors = np.concatenate([
            self.Vt.T,
            self.problem_features
        ], axis=1)

        # Calculate cosine similarity
        reference_vector = problem_vectors[problem_idx]
        similarities = np.dot(problem_vectors, reference_vector) / (
                np.linalg.norm(problem_vectors, axis=1) * np.linalg.norm(reference_vector)
        )

        # Get top N similar problems (excluding the reference problem)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar + 1]

        return [(self.problem_ids[idx], similarities[idx]) for idx in similar_indices]