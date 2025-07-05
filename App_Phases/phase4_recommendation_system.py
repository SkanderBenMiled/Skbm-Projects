"""
Phase 4: Music Recommendation System
Music Recommendation System - Spotify Dataset

This phase implements a comprehensive recommendation system using song features,
clustering results, and the Spotify API integration. The system calculates
similarity between songs and recommends the most similar tracks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import json
import pickle
from pathlib import Path
from datetime import datetime
import streamlit as st

# Machine Learning imports
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats

# Spotify API integration
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    print("⚠️ Spotipy not available. Install with: pip install spotipy")

# Suppress warnings
warnings.filterwarnings('ignore')

class SpotifyRecommendationSystem:
    """
    Advanced Music Recommendation System using Spotify Dataset
    """
    
    def __init__(self, data_path="Djeezy/Spotify_Data"):
        """
        Initialize the recommendation system
        
        Args:
            data_path (str): Path to the Spotify data folder
        """
        self.data_path = Path(data_path)
        self.data = None
        self.data_with_genres = None
        self.audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                              'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        self.scaler = StandardScaler()
        self.recommendation_models = {}
        self.cluster_model = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.spotify_client = None
        
        # Initialize Spotify API if available
        if SPOTIPY_AVAILABLE:
            self._init_spotify_client()
    
    def _init_spotify_client(self):
        """
        Initialize Spotify API client
        """
        try:
            # You would need to set your Spotify API credentials
            # For demo purposes, we'll create a placeholder
            # client_credentials_manager = SpotifyClientCredentials(
            #     client_id="YOUR_CLIENT_ID",
            #     client_secret="YOUR_CLIENT_SECRET"
            # )
            # self.spotify_client = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            print("🎵 Spotify API client initialized (placeholder)")
        except Exception as e:
            print(f"⚠️ Spotify API initialization failed: {e}")
    
    def load_data(self):
        """
        Load datasets and previous analysis results
        """
        print("🔄 Loading data for recommendation system...")
        
        # Load main dataset
        try:
            self.data = pd.read_csv(self.data_path / "data.csv")
            print(f"✅ Main dataset loaded: {self.data.shape[0]:,} tracks")
        except Exception as e:
            print(f"❌ Error loading main dataset: {e}")
            return False
            
        # Load genre dataset if available
        try:
            self.data_with_genres = pd.read_csv(self.data_path / "data_w_genres.csv")
            print(f"✅ Genre dataset loaded: {self.data_with_genres.shape[0]:,} records")
        except Exception as e:
            print(f"⚠️ Genre dataset not available: {e}")
        
        # Load clustering results if available
        try:
            if Path("cluster_profiles.json").exists():
                with open("cluster_profiles.json", 'r') as f:
                    self.cluster_profiles = json.load(f)
                print(f"✅ Cluster profiles loaded: {len(self.cluster_profiles)} clusters")
            else:
                print("⚠️ No cluster profiles found. Run Phase 3 first for better recommendations.")
        except Exception as e:
            print(f"⚠️ Error loading cluster profiles: {e}")
            
        return True
    
    def prepare_recommendation_data(self):
        """
        Prepare data for recommendation system
        """
        print("\n🔧 PREPARING RECOMMENDATION DATA")
        print("=" * 50)
        
        if self.data is None:
            print("❌ No data loaded")
            return False
        
        # Select available audio features
        available_features = [f for f in self.audio_features if f in self.data.columns]
        
        if len(available_features) < 3:
            print("❌ Insufficient audio features for recommendations")
            return False
        
        print(f"📊 Using {len(available_features)} features for recommendations:")
        for feature in available_features:
            print(f"   - {feature}")
        
        # Prepare feature matrix
        feature_data = self.data[available_features + ['popularity']].copy()
        
        # Handle missing values
        print(f"\n🔍 Data preprocessing:")
        print(f"   Original shape: {feature_data.shape}")
        print(f"   Missing values: {feature_data.isnull().sum().sum()}")
        
        # Remove rows with missing values
        feature_data = feature_data.dropna()
        self.cleaned_data = self.data.loc[feature_data.index].copy()
        
        print(f"   Final shape: {feature_data.shape}")
        
        # Scale features
        self.feature_matrix = self.scaler.fit_transform(feature_data[available_features])
        
        # Store feature names and indices
        self.feature_names = available_features
        self.track_indices = feature_data.index.tolist()
        
        print(f"✅ Feature matrix prepared: {self.feature_matrix.shape}")
        
        return True
    
    def build_similarity_matrix(self, similarity_method='cosine'):
        """
        Build similarity matrix between all tracks
        
        Args:
            similarity_method (str): Method to calculate similarity ('cosine', 'euclidean')
        """
        print(f"\n🔄 BUILDING SIMILARITY MATRIX ({similarity_method})")
        print("=" * 50)
        
        if self.feature_matrix is None:
            print("❌ Feature matrix not prepared")
            return False
        
        print(f"📊 Calculating {similarity_method} similarity for {len(self.feature_matrix):,} tracks...")
        
        if similarity_method == 'cosine':
            # Calculate cosine similarity
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
        elif similarity_method == 'euclidean':
            # Calculate euclidean distance and convert to similarity
            distances = euclidean_distances(self.feature_matrix)
            # Convert distances to similarities (higher = more similar)
            self.similarity_matrix = 1 / (1 + distances)
        else:
            print(f"❌ Unknown similarity method: {similarity_method}")
            return False
        
        print(f"✅ Similarity matrix built: {self.similarity_matrix.shape}")
        
        # Save similarity matrix
        similarity_file = f"similarity_matrix_{similarity_method}.pkl"
        with open(similarity_file, 'wb') as f:
            pickle.dump(self.similarity_matrix, f)
        
        print(f"💾 Similarity matrix saved as '{similarity_file}'")
        
        return True
    
    def build_cluster_based_recommender(self):
        """
        Build cluster-based recommendation system
        """
        print("\n🎯 BUILDING CLUSTER-BASED RECOMMENDER")
        print("=" * 50)
        
        if self.feature_matrix is None:
            print("❌ Feature matrix not prepared")
            return False
        
        # Perform clustering for recommendations
        print("🔄 Performing clustering for recommendations...")
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters_for_recommendations()
        
        # Build K-means model
        self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(self.feature_matrix)
        
        # Add cluster labels to cleaned data
        self.cleaned_data['cluster'] = cluster_labels
        
        print(f"✅ Cluster-based recommender built with {optimal_k} clusters")
        
        # Analyze cluster characteristics
        self._analyze_recommendation_clusters()
        
        return True
    
    def _find_optimal_clusters_for_recommendations(self, max_k=20):
        """
        Find optimal number of clusters for recommendations
        """
        print("🔍 Finding optimal clusters for recommendations...")
        
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(self.feature_matrix) // 10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.feature_matrix)
            
            from sklearn.metrics import silhouette_score
            score = silhouette_score(self.feature_matrix, labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"   Optimal k for recommendations: {optimal_k}")
        
        return optimal_k
    
    def _analyze_recommendation_clusters(self):
        """
        Analyze clusters created for recommendations
        """
        print("📊 Analyzing recommendation clusters...")
        
        cluster_stats = self.cleaned_data.groupby('cluster').agg({
            **{feature: 'mean' for feature in self.feature_names},
            'popularity': ['mean', 'count']
        }).round(3)
        
        print("Cluster characteristics:")
        print(cluster_stats.head(10))
        
        return cluster_stats
    
    def build_knn_recommender(self, n_neighbors=50):
        """
        Build K-Nearest Neighbors recommender
        
        Args:
            n_neighbors (int): Number of neighbors to consider
        """
        print(f"\n🎯 BUILDING KNN RECOMMENDER (k={n_neighbors})")
        print("=" * 50)
        
        if self.feature_matrix is None:
            print("❌ Feature matrix not prepared")
            return False
        
        # Build KNN model
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, 
                                         metric='cosine', 
                                         algorithm='brute')
        self.knn_model.fit(self.feature_matrix)
        
        print(f"✅ KNN recommender built with {n_neighbors} neighbors")
        
        return True
    
    def get_recommendations(self, track_name=None, track_id=None, track_features=None, 
                          method='hybrid', n_recommendations=10):
        """
        Get music recommendations
        
        Args:
            track_name (str): Name of the track to get recommendations for
            track_id (str): Spotify ID of the track
            track_features (dict): Audio features of the track
            method (str): Recommendation method ('similarity', 'cluster', 'knn', 'hybrid')
            n_recommendations (int): Number of recommendations to return
        """
        print(f"\n🎵 GETTING RECOMMENDATIONS ({method})")
        print("=" * 50)
        
        # Find track index
        track_idx = None
        
        if track_name:
            # Find track by name
            matching_tracks = self.cleaned_data[
                self.cleaned_data['name'].str.contains(track_name, case=False, na=False)
            ]
            
            if len(matching_tracks) == 0:
                print(f"❌ Track '{track_name}' not found")
                return []
            
            track_idx = matching_tracks.index[0]
            print(f"🎵 Found track: {matching_tracks.iloc[0]['name']} by {matching_tracks.iloc[0]['artists']}")
        
        elif track_features:
            # Find similar track based on features
            track_idx = self._find_similar_track_by_features(track_features)
        
        if track_idx is None:
            print("❌ Could not identify track for recommendations")
            return []
        
        # Get recommendations based on method
        if method == 'similarity':
            recommendations = self._get_similarity_recommendations(track_idx, n_recommendations)
        elif method == 'cluster':
            recommendations = self._get_cluster_recommendations(track_idx, n_recommendations)
        elif method == 'knn':
            recommendations = self._get_knn_recommendations(track_idx, n_recommendations)
        elif method == 'hybrid':
            recommendations = self._get_hybrid_recommendations(track_idx, n_recommendations)
        else:
            print(f"❌ Unknown recommendation method: {method}")
            return []
        
        # Format recommendations
        formatted_recommendations = self._format_recommendations(recommendations, track_idx)
        
        return formatted_recommendations
    
    def _get_similarity_recommendations(self, track_idx, n_recommendations):
        """
        Get recommendations using similarity matrix
        """
        if self.similarity_matrix is None:
            print("❌ Similarity matrix not built")
            return []
        
        # Get track position in similarity matrix
        track_pos = self.track_indices.index(track_idx)
        
        # Get similarity scores
        similarity_scores = self.similarity_matrix[track_pos]
        
        # Get top similar tracks (excluding the track itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        # Convert back to original indices
        recommendations = [self.track_indices[i] for i in similar_indices]
        
        return recommendations
    
    def _get_cluster_recommendations(self, track_idx, n_recommendations):
        """
        Get recommendations using cluster-based approach
        """
        if self.cluster_model is None:
            print("❌ Cluster model not built")
            return []
        
        # Get track cluster
        track_cluster = self.cleaned_data.loc[track_idx, 'cluster']
        
        # Get all tracks in the same cluster
        cluster_tracks = self.cleaned_data[self.cleaned_data['cluster'] == track_cluster]
        
        # Remove the input track
        cluster_tracks = cluster_tracks[cluster_tracks.index != track_idx]
        
        # Sort by popularity and get top recommendations
        recommendations = cluster_tracks.sort_values('popularity', ascending=False).head(n_recommendations)
        
        return recommendations.index.tolist()
    
    def _get_knn_recommendations(self, track_idx, n_recommendations):
        """
        Get recommendations using KNN approach
        """
        if not hasattr(self, 'knn_model'):
            print("❌ KNN model not built")
            return []
        
        # Get track position in feature matrix
        track_pos = self.track_indices.index(track_idx)
        
        # Get track features
        track_features = self.feature_matrix[track_pos].reshape(1, -1)
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors(track_features)
        
        # Get recommendations (excluding the track itself)
        neighbor_indices = indices[0][1:n_recommendations+1]
        
        # Convert back to original indices
        recommendations = [self.track_indices[i] for i in neighbor_indices]
        
        return recommendations
    
    def _get_hybrid_recommendations(self, track_idx, n_recommendations):
        """
        Get recommendations using hybrid approach
        """
        # Combine different methods
        similarity_recs = self._get_similarity_recommendations(track_idx, n_recommendations // 2)
        cluster_recs = self._get_cluster_recommendations(track_idx, n_recommendations // 2)
        
        # Combine and remove duplicates
        all_recs = similarity_recs + cluster_recs
        unique_recs = list(dict.fromkeys(all_recs))  # Remove duplicates while preserving order
        
        return unique_recs[:n_recommendations]
    
    def _find_similar_track_by_features(self, target_features):
        """
        Find the most similar track based on audio features
        """
        # Create feature vector
        feature_vector = np.array([target_features.get(f, 0) for f in self.feature_names])
        
        # Scale the features
        scaled_features = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Find most similar track
        similarities = cosine_similarity(scaled_features, self.feature_matrix)[0]
        most_similar_idx = np.argmax(similarities)
        
        return self.track_indices[most_similar_idx]
    
    def _format_recommendations(self, recommendations, original_track_idx):
        """
        Format recommendations for output
        """
        formatted_recs = []
        
        # Get original track info
        original_track = self.cleaned_data.loc[original_track_idx]
        
        print(f"📋 Recommendations for: {original_track['name']} by {original_track['artists']}")
        print("-" * 60)
        
        for i, rec_idx in enumerate(recommendations, 1):
            track = self.cleaned_data.loc[rec_idx]
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity_score(original_track_idx, rec_idx)
            
            rec_info = {
                'rank': i,
                'name': track['name'],
                'artist': track['artists'],
                'popularity': track['popularity'],
                'year': track.get('year', 'Unknown'),
                'similarity_score': similarity_score,
                'spotify_id': track.get('id', ''),
                'audio_features': {feature: track[feature] for feature in self.feature_names if feature in track}
            }
            
            formatted_recs.append(rec_info)
            
            print(f"   {i:2d}. {track['name']} by {track['artists']}")
            print(f"       Popularity: {track['popularity']:.0f} | "
                  f"Similarity: {similarity_score:.3f} | "
                  f"Year: {track.get('year', 'Unknown')}")
        
        return formatted_recs
    
    def _calculate_similarity_score(self, track_idx1, track_idx2):
        """
        Calculate similarity score between two tracks
        """
        if self.similarity_matrix is not None:
            pos1 = self.track_indices.index(track_idx1)
            pos2 = self.track_indices.index(track_idx2)
            return self.similarity_matrix[pos1][pos2]
        else:
            # Calculate on-the-fly
            features1 = self.feature_matrix[self.track_indices.index(track_idx1)]
            features2 = self.feature_matrix[self.track_indices.index(track_idx2)]
            
            return cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]
    
    def analyze_recommendation_quality(self, n_test_tracks=100):
        """
        Analyze the quality of recommendations
        """
        print("\n📊 ANALYZING RECOMMENDATION QUALITY")
        print("=" * 50)
        
        if len(self.cleaned_data) < n_test_tracks:
            n_test_tracks = len(self.cleaned_data) // 2
        
        print(f"🔄 Testing with {n_test_tracks} random tracks...")
        
        # Sample random tracks for testing
        test_tracks = self.cleaned_data.sample(n=n_test_tracks, random_state=42)
        
        quality_metrics = {
            'avg_similarity_score': [],
            'avg_popularity_score': [],
            'genre_diversity': [],
            'year_diversity': []
        }
        
        for idx in test_tracks.index:
            # Get recommendations
            recommendations = self.get_recommendations(
                track_name=test_tracks.loc[idx, 'name'],
                method='hybrid',
                n_recommendations=10
            )
            
            if recommendations:
                # Calculate metrics
                similarity_scores = [rec['similarity_score'] for rec in recommendations]
                popularity_scores = [rec['popularity'] for rec in recommendations]
                years = [rec['year'] for rec in recommendations if rec['year'] != 'Unknown']
                
                quality_metrics['avg_similarity_score'].append(np.mean(similarity_scores))
                quality_metrics['avg_popularity_score'].append(np.mean(popularity_scores))
                quality_metrics['year_diversity'].append(len(set(years)) if years else 0)
        
        # Calculate overall quality metrics
        results = {
            'avg_similarity': np.mean(quality_metrics['avg_similarity_score']),
            'avg_popularity': np.mean(quality_metrics['avg_popularity_score']),
            'avg_year_diversity': np.mean(quality_metrics['year_diversity']),
            'recommendation_success_rate': len(quality_metrics['avg_similarity_score']) / n_test_tracks
        }
        
        print(f"📈 Quality Analysis Results:")
        print(f"   Average Similarity Score: {results['avg_similarity']:.3f}")
        print(f"   Average Popularity Score: {results['avg_popularity']:.1f}")
        print(f"   Average Year Diversity: {results['avg_year_diversity']:.1f}")
        print(f"   Recommendation Success Rate: {results['recommendation_success_rate']:.1%}")
        
        return results
    
    def create_recommendation_dashboard(self):
        """
        Create interactive dashboard for the recommendation system
        """
        print("\n🎨 CREATING RECOMMENDATION DASHBOARD")
        print("=" * 50)
        
        # Create sample recommendations for visualization
        sample_tracks = self.cleaned_data.sample(n=5, random_state=42)
        
        # Create dashboard using Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Cluster Analysis', 
                           'Popularity vs Similarity', 'Recommendation Network'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Feature distribution
        for i, feature in enumerate(self.feature_names[:3]):
            fig.add_trace(
                go.Histogram(x=self.cleaned_data[feature], name=feature, opacity=0.7),
                row=1, col=1
            )
        
        # 2. Cluster analysis
        if 'cluster' in self.cleaned_data.columns:
            fig.add_trace(
                go.Scatter(x=self.cleaned_data['energy'], y=self.cleaned_data['danceability'],
                          mode='markers', 
                          marker=dict(color=self.cleaned_data['cluster'], 
                                    colorscale='viridis', showscale=True),
                          name='Tracks by Cluster'),
                row=1, col=2
            )
        
        # 3. Popularity vs Similarity analysis
        if len(sample_tracks) > 1:
            sample_idx = sample_tracks.index[0]
            recommendations = self.get_recommendations(
                track_name=sample_tracks.iloc[0]['name'],
                method='hybrid',
                n_recommendations=20
            )
            
            if recommendations:
                similarities = [rec['similarity_score'] for rec in recommendations]
                popularities = [rec['popularity'] for rec in recommendations]
                
                fig.add_trace(
                    go.Scatter(x=similarities, y=popularities, mode='markers',
                              name='Recommendations', marker=dict(size=10)),
                    row=2, col=1
                )
        
        # 4. Feature importance
        if hasattr(self, 'feature_importance'):
            fig.add_trace(
                go.Bar(x=self.feature_names, y=self.feature_importance, 
                      name='Feature Importance'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Music Recommendation System Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save dashboard
        fig.write_html("recommendation_dashboard.html")
        print("✅ Dashboard saved as 'recommendation_dashboard.html'")
        
        fig.show()
        
        return fig
    
    def save_recommendation_system(self, filename="spotify_recommendation_system.pkl"):
        """
        Save the complete recommendation system
        """
        print(f"\n💾 SAVING RECOMMENDATION SYSTEM")
        print("=" * 50)
        
        # Prepare system data
        system_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'track_indices': self.track_indices,
            'cluster_model': self.cluster_model,
            'knn_model': getattr(self, 'knn_model', None),
            'similarity_matrix': self.similarity_matrix,
            'cleaned_data': self.cleaned_data,
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'n_tracks': len(self.cleaned_data),
                'n_features': len(self.feature_names),
                'model_version': '1.0'
            }
        }
        
        # Save system
        with open(filename, 'wb') as f:
            pickle.dump(system_data, f)
        
        print(f"✅ Recommendation system saved as '{filename}'")
        print(f"📊 System info:")
        print(f"   Tracks: {len(self.cleaned_data):,}")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Models: Clustering, KNN, Similarity")
        
        return True
    
    def load_recommendation_system(self, filename="spotify_recommendation_system.pkl"):
        """
        Load a saved recommendation system
        """
        print(f"\n📂 LOADING RECOMMENDATION SYSTEM")
        print("=" * 50)
        
        try:
            with open(filename, 'rb') as f:
                system_data = pickle.load(f)
            
            # Restore system components
            self.scaler = system_data['scaler']
            self.feature_names = system_data['feature_names']
            self.track_indices = system_data['track_indices']
            self.cluster_model = system_data['cluster_model']
            self.knn_model = system_data.get('knn_model')
            self.similarity_matrix = system_data['similarity_matrix']
            self.cleaned_data = system_data['cleaned_data']
            
            # Rebuild feature matrix
            self.feature_matrix = self.scaler.transform(
                self.cleaned_data[self.feature_names]
            )
            
            print(f"✅ Recommendation system loaded successfully")
            print(f"📊 System info:")
            print(f"   Tracks: {len(self.cleaned_data):,}")
            print(f"   Features: {len(self.feature_names)}")
            print(f"   Created: {system_data['metadata']['created_date']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading recommendation system: {e}")
            return False
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive recommendation system report
        """
        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)
        
        report = {
            'system_overview': {
                'total_tracks': len(self.cleaned_data) if self.cleaned_data is not None else 0,
                'audio_features': self.feature_names,
                'recommendation_methods': ['similarity', 'cluster', 'knn', 'hybrid'],
                'models_built': []
            },
            'performance_metrics': {},
            'system_capabilities': [],
            'usage_examples': [],
            'recommendations_for_improvement': []
        }
        
        # Fill system overview
        if self.similarity_matrix is not None:
            report['system_overview']['models_built'].append('Similarity Matrix')
        if self.cluster_model is not None:
            report['system_overview']['models_built'].append('Cluster Model')
        if hasattr(self, 'knn_model'):
            report['system_overview']['models_built'].append('KNN Model')
        
        # System capabilities
        report['system_capabilities'] = [
            "Content-based filtering using audio features",
            "Clustering-based recommendations",
            "K-nearest neighbors recommendations",
            "Hybrid recommendation approach",
            "Similarity-based track matching",
            "Interactive dashboard visualization",
            "Batch recommendation processing",
            "System persistence and loading"
        ]
        
        # Usage examples
        report['usage_examples'] = [
            "Get recommendations for a specific track by name",
            "Find similar tracks based on audio features",
            "Discover tracks within the same musical cluster",
            "Generate hybrid recommendations combining multiple methods",
            "Analyze recommendation quality and diversity"
        ]
        
        # Recommendations for improvement
        report['recommendations_for_improvement'] = [
            "Integrate with Spotify API for real-time data",
            "Add collaborative filtering using user behavior",
            "Implement deep learning models for better accuracy",
            "Add genre-based filtering and recommendations",
            "Include user feedback mechanism",
            "Develop mobile application interface",
            "Add real-time model updating capabilities"
        ]
        
        # Save report
        with open('recommendation_system_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Comprehensive report saved as 'recommendation_system_report.json'")
        
        return report


def main():
    """
    Main function to execute Phase 4 - Recommendation System
    """
    print("🎵 SPOTIFY MUSIC RECOMMENDATION SYSTEM - PHASE 4")
    print("=" * 70)
    print("Phase 4: Advanced Music Recommendation System")
    print("=" * 70)
    
    # Initialize recommendation system
    recommender = SpotifyRecommendationSystem()
    
    # Load data
    if not recommender.load_data():
        print("❌ Failed to load data")
        return
    
    results = {}
    
    # 1. Prepare data for recommendations
    print("\n" + "="*50)
    print("1. DATA PREPARATION")
    print("="*50)
    if not recommender.prepare_recommendation_data():
        print("❌ Failed to prepare data")
        return
    
    # 2. Build similarity matrix
    print("\n" + "="*50)
    print("2. SIMILARITY MATRIX CONSTRUCTION")
    print("="*50)
    recommender.build_similarity_matrix(similarity_method='cosine')
    
    # 3. Build cluster-based recommender
    print("\n" + "="*50)
    print("3. CLUSTER-BASED RECOMMENDER")
    print("="*50)
    recommender.build_cluster_based_recommender()
    
    # 4. Build KNN recommender
    print("\n" + "="*50)
    print("4. KNN RECOMMENDER")
    print("="*50)
    recommender.build_knn_recommender()
    
    # 5. Test recommendations
    print("\n" + "="*50)
    print("5. TESTING RECOMMENDATIONS")
    print("="*50)
    
    # Test with sample tracks
    sample_tracks = recommender.cleaned_data.sample(n=3, random_state=42)
    
    for idx, track in sample_tracks.iterrows():
        print(f"\n🎵 Testing recommendations for: {track['name']} by {track['artists']}")
        print("-" * 60)
        
        # Test different methods
        methods = ['similarity', 'cluster', 'knn', 'hybrid']
        for method in methods:
            print(f"\n📊 {method.upper()} Method:")
            recommendations = recommender.get_recommendations(
                track_name=track['name'],
                method=method,
                n_recommendations=5
            )
            results[f"{method}_recommendations"] = recommendations
    
    # 6. Analyze recommendation quality
    print("\n" + "="*50)
    print("6. RECOMMENDATION QUALITY ANALYSIS")
    print("="*50)
    quality_results = recommender.analyze_recommendation_quality()
    results['quality_analysis'] = quality_results
    
    # 7. Create dashboard
    print("\n" + "="*50)
    print("7. INTERACTIVE DASHBOARD")
    print("="*50)
    dashboard = recommender.create_recommendation_dashboard()
    results['dashboard'] = dashboard
    
    # 8. Save system
    print("\n" + "="*50)
    print("8. SAVING SYSTEM")
    print("="*50)
    recommender.save_recommendation_system()
    
    # 9. Generate comprehensive report
    print("\n" + "="*50)
    print("9. COMPREHENSIVE REPORT")
    print("="*50)
    report = recommender.generate_comprehensive_report()
    results['report'] = report
    
    # Summary
    print("\n✅ Phase 4 Recommendation System completed successfully!")
    print("📊 System Summary:")
    print(f"   🎵 Total tracks: {len(recommender.cleaned_data):,}")
    print(f"   🎼 Audio features: {len(recommender.feature_names)}")
    print(f"   🎯 Recommendation methods: 4 (similarity, cluster, knn, hybrid)")
    print(f"   📈 Average similarity score: {quality_results['avg_similarity']:.3f}")
    print(f"   ⭐ Average popularity score: {quality_results['avg_popularity']:.1f}")
    print(f"   ✅ Success rate: {quality_results['recommendation_success_rate']:.1%}")
    
    print("\n📁 Files generated:")
    print("   - similarity_matrix_cosine.pkl")
    print("   - spotify_recommendation_system.pkl")
    print("   - recommendation_dashboard.html")
    print("   - recommendation_system_report.json")
    
    print("\n🎉 Complete Music Recommendation System Built!")
    print("🔧 System is ready for production use!")
    
    return recommender, results


def create_streamlit_app():
    """
    Create a Streamlit web application for the recommendation system
    """
    st.title("🎵 Spotify Music Recommendation System")
    st.markdown("### Discover your next favorite song!")
    
    # Initialize session state
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    
    # Sidebar for system loading
    with st.sidebar:
        st.header("🔧 System Controls")
        
        if st.button("Load Recommendation System"):
            with st.spinner("Loading recommendation system..."):
                recommender = SpotifyRecommendationSystem()
                if recommender.load_data() and recommender.prepare_recommendation_data():
                    st.session_state.recommender = recommender
                    st.success("System loaded successfully!")
                else:
                    st.error("Failed to load system")
    
    # Main interface
    if st.session_state.recommender is not None:
        recommender = st.session_state.recommender
        
        # Input methods
        input_method = st.selectbox(
            "How would you like to get recommendations?",
            ["Search by Song Name", "Input Audio Features"]
        )
        
        if input_method == "Search by Song Name":
            # Song search
            song_name = st.text_input("Enter song name:")
            
            if song_name:
                # Find matching songs
                matches = recommender.cleaned_data[
                    recommender.cleaned_data['name'].str.contains(song_name, case=False, na=False)
                ].head(10)
                
                if len(matches) > 0:
                    # Display matches
                    selected_song = st.selectbox(
                        "Select a song:",
                        options=matches.index,
                        format_func=lambda x: f"{matches.loc[x, 'name']} by {matches.loc[x, 'artists']}"
                    )
                    
                    # Recommendation parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        method = st.selectbox(
                            "Recommendation method:",
                            ["hybrid", "similarity", "cluster", "knn"]
                        )
                    with col2:
                        n_recs = st.slider("Number of recommendations:", 1, 20, 10)
                    
                    # Get recommendations
                    if st.button("Get Recommendations"):
                        with st.spinner("Generating recommendations..."):
                            recommendations = recommender.get_recommendations(
                                track_name=matches.loc[selected_song, 'name'],
                                method=method,
                                n_recommendations=n_recs
                            )
                            
                            if recommendations:
                                st.success(f"Found {len(recommendations)} recommendations!")
                                
                                # Display recommendations
                                for i, rec in enumerate(recommendations):
                                    with st.expander(f"{i+1}. {rec['name']} by {rec['artist']}"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**Popularity:** {rec['popularity']:.0f}")
                                            st.write(f"**Year:** {rec['year']}")
                                            st.write(f"**Similarity:** {rec['similarity_score']:.3f}")
                                        with col2:
                                            # Audio features
                                            features = rec['audio_features']
                                            for feature, value in features.items():
                                                st.write(f"**{feature.capitalize()}:** {value:.3f}")
                            else:
                                st.error("No recommendations found")
                else:
                    st.warning("No matching songs found")
        
        elif input_method == "Input Audio Features":
            # Audio features input
            st.subheader("Audio Features")
            
            features = {}
            col1, col2 = st.columns(2)
            
            with col1:
                features['energy'] = st.slider("Energy", 0.0, 1.0, 0.5)
                features['danceability'] = st.slider("Danceability", 0.0, 1.0, 0.5)
                features['valence'] = st.slider("Valence (Positivity)", 0.0, 1.0, 0.5)
                features['acousticness'] = st.slider("Acousticness", 0.0, 1.0, 0.5)
            
            with col2:
                features['instrumentalness'] = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
                features['liveness'] = st.slider("Liveness", 0.0, 1.0, 0.5)
                features['speechiness'] = st.slider("Speechiness", 0.0, 1.0, 0.5)
                features['tempo'] = st.slider("Tempo", 0.0, 200.0, 120.0)
            
            # Get recommendations
            if st.button("Find Similar Songs"):
                with st.spinner("Finding similar songs..."):
                    recommendations = recommender.get_recommendations(
                        track_features=features,
                        method='hybrid',
                        n_recommendations=10
                    )
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} similar songs!")
                        
                        # Display recommendations
                        for i, rec in enumerate(recommendations):
                            with st.expander(f"{i+1}. {rec['name']} by {rec['artist']}"):
                                st.write(f"**Popularity:** {rec['popularity']:.0f}")
                                st.write(f"**Year:** {rec['year']}")
                                st.write(f"**Similarity:** {rec['similarity_score']:.3f}")
                    else:
                        st.error("No similar songs found")
        
        # System statistics
        with st.expander("📊 System Statistics"):
            st.write(f"**Total Tracks:** {len(recommender.cleaned_data):,}")
            st.write(f"**Audio Features:** {len(recommender.feature_names)}")
            st.write(f"**Recommendation Methods:** 4")
            
            # Feature distribution
            if len(recommender.feature_names) > 0:
                feature_to_plot = st.selectbox(
                    "Select feature to visualize:",
                    recommender.feature_names
                )
                
                fig = px.histogram(
                    recommender.cleaned_data,
                    x=feature_to_plot,
                    title=f"Distribution of {feature_to_plot.capitalize()}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please load the recommendation system first using the sidebar.")


if __name__ == "__main__":
    # Check if running in Streamlit
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        create_streamlit_app()
    else:
        # Execute Phase 4
        recommender, results = main()
        
        # Demo recommendations
        print("\n" + "="*50)
        print("🎯 DEMO RECOMMENDATIONS")
        print("="*50)
        
        # Get some popular tracks for demo
        popular_tracks = recommender.cleaned_data.nlargest(5, 'popularity')
        
        print("🎵 Try these popular tracks for recommendations:")
        for idx, track in popular_tracks.iterrows():
            print(f"   - {track['name']} by {track['artists']} (Popularity: {track['popularity']:.0f})")
        
        print("\n💡 Usage Examples:")
        print("   # Get recommendations for a specific track")
        print("   recommendations = recommender.get_recommendations(")
        print("       track_name='Shape of You',")
        print("       method='hybrid',")
        print("       n_recommendations=10")
        print("   )")
        print("\n   # Get recommendations based on audio features")
        print("   recommendations = recommender.get_recommendations(")
        print("       track_features={'energy': 0.8, 'danceability': 0.7, 'valence': 0.6},")
        print("       method='similarity',")
        print("       n_recommendations=5")
        print("   )")
        
        print("\n🌐 Web Interface:")
        print("   Run: streamlit run phase4_recommendation_system.py streamlit")
        print("   This will launch an interactive web interface!")
        
        print("\n🎉 Complete Music Recommendation System Ready!")
        print("🚀 All 4 phases completed successfully!")
