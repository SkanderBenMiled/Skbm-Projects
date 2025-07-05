"""
Phase 3: Clustering Analysis
Music Recommendation System - Spotify Dataset

This phase uses clustering techniques to group genres and songs based on their characteristics.
Clustering helps in discovering patterns and similarities within the dataset, enabling us to
gain insights into the structure of the music data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from collections import Counter
import ast
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpotifyClustering:
    """
    Advanced Clustering Analysis for Spotify Dataset
    """
    
    def __init__(self, data_path="Djeezy/Spotify_Data"):
        """
        Initialize the clustering analysis
        
        Args:
            data_path (str): Path to the Spotify data folder
        """
        self.data_path = Path(data_path)
        self.data = None
        self.data_with_genres = None
        self.audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                              'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        self.scaler = StandardScaler()
        self.cluster_results = {}
        
    def load_data(self):
        """
        Load datasets for clustering analysis
        """
        print("🔄 Loading datasets for clustering analysis...")
        
        try:
            self.data = pd.read_csv(self.data_path / "data.csv")
            print(f"✅ Main dataset loaded: {self.data.shape[0]:,} tracks")
        except Exception as e:
            print(f"❌ Error loading main dataset: {e}")
            return False
            
        try:
            self.data_with_genres = pd.read_csv(self.data_path / "data_w_genres.csv")
            print(f"✅ Genre dataset loaded: {self.data_with_genres.shape[0]:,} records")
        except Exception as e:
            print(f"⚠️ Genre dataset not available: {e}")
            
        return True
    
    def prepare_clustering_data(self):
        """
        Prepare data for clustering analysis
        """
        print("\n🔧 PREPARING DATA FOR CLUSTERING")
        print("=" * 50)
        
        if self.data is None:
            print("❌ No data loaded")
            return None
            
        # Select available audio features
        available_features = [f for f in self.audio_features if f in self.data.columns]
        
        if len(available_features) < 3:
            print("❌ Insufficient audio features for clustering")
            return None
            
        print(f"📊 Using {len(available_features)} audio features for clustering:")
        for feature in available_features:
            print(f"   - {feature}")
            
        # Prepare clustering dataset
        clustering_data = self.data[available_features + ['popularity']].copy()
        
        # Handle missing values
        print(f"\n🔍 Data Quality Check:")
        print(f"   Original shape: {clustering_data.shape}")
        print(f"   Missing values: {clustering_data.isnull().sum().sum()}")
        
        # Remove rows with missing values
        clustering_data = clustering_data.dropna()
        print(f"   Final shape: {clustering_data.shape}")
        
        # Feature scaling
        feature_data = clustering_data[available_features]
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Create scaled DataFrame
        scaled_df = pd.DataFrame(scaled_features, columns=available_features, index=clustering_data.index)
        scaled_df['popularity'] = clustering_data['popularity'].values
        
        print(f"✅ Data prepared for clustering: {scaled_df.shape[0]:,} samples")
        
        return scaled_df, available_features
    
    def find_optimal_clusters(self, data, max_clusters=15):
        """
        Find optimal number of clusters using multiple methods
        """
        print("\n🔍 FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("=" * 50)
        
        # Methods to evaluate
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        davies_bouldin_scores = []
        
        k_range = range(2, max_clusters + 1)
        
        print("🔄 Evaluating different cluster numbers...")
        
        for k in k_range:
            # K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(data, cluster_labels))
            davies_bouldin_scores.append(davies_bouldin_score(data, cluster_labels))
            
            if k % 3 == 0:
                print(f"   k={k}: Silhouette={silhouette_scores[-1]:.3f}, Calinski={calinski_scores[-1]:.1f}")
        
        # Find optimal k using different methods
        # Elbow method (look for elbow in inertia)
        elbow_k = self._find_elbow_point(list(k_range), inertias)
        
        # Best silhouette score
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        # Best Calinski-Harabasz score
        best_calinski_k = k_range[np.argmax(calinski_scores)]
        
        # Best Davies-Bouldin score (lower is better)
        best_db_k = k_range[np.argmin(davies_bouldin_scores)]
        
        print(f"\n📊 Optimal k suggestions:")
        print(f"   Elbow method: k={elbow_k}")
        print(f"   Silhouette score: k={best_silhouette_k} (score: {max(silhouette_scores):.3f})")
        print(f"   Calinski-Harabasz: k={best_calinski_k} (score: {max(calinski_scores):.1f})")
        print(f"   Davies-Bouldin: k={best_db_k} (score: {min(davies_bouldin_scores):.3f})")
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Clustering Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # Elbow method
        axes[0, 0].plot(k_range, inertias, 'bo-')
        axes[0, 0].axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow k={elbow_k}')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[0, 1].plot(k_range, silhouette_scores, 'go-')
        axes[0, 1].axvline(x=best_silhouette_k, color='red', linestyle='--', label=f'Best k={best_silhouette_k}')
        axes[0, 1].set_title('Silhouette Score')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz scores
        axes[1, 0].plot(k_range, calinski_scores, 'mo-')
        axes[1, 0].axvline(x=best_calinski_k, color='red', linestyle='--', label=f'Best k={best_calinski_k}')
        axes[1, 0].set_title('Calinski-Harabasz Score')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin scores
        axes[1, 1].plot(k_range, davies_bouldin_scores, 'co-')
        axes[1, 1].axvline(x=best_db_k, color='red', linestyle='--', label=f'Best k={best_db_k}')
        axes[1, 1].set_title('Davies-Bouldin Score')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Davies-Bouldin Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('clustering_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Consensus optimal k (most common suggestion)
        suggestions = [elbow_k, best_silhouette_k, best_calinski_k, best_db_k]
        optimal_k = Counter(suggestions).most_common(1)[0][0]
        
        print(f"\n🎯 Consensus optimal k: {optimal_k}")
        
        return {
            'optimal_k': optimal_k,
            'elbow_k': elbow_k,
            'silhouette_k': best_silhouette_k,
            'calinski_k': best_calinski_k,
            'davies_bouldin_k': best_db_k,
            'metrics': {
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores
            }
        }
    
    def _find_elbow_point(self, x, y):
        """
        Find elbow point in a curve using the distance method
        """
        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Calculate distances from each point to the line connecting first and last points
        distances = []
        
        # Line from first to last point
        line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        
        for i in range(len(x)):
            point = np.array([x[i], y[i]])
            first_point = np.array([x[0], y[0]])
            
            # Vector from first point to current point
            point_vec = point - first_point
            
            # Distance from point to line
            distance = np.linalg.norm(point_vec - np.dot(point_vec, line_vec_norm) * line_vec_norm)
            distances.append(distance)
        
        # Return x value of point with maximum distance
        return x[np.argmax(distances)]
    
    def perform_kmeans_clustering(self, data, available_features, n_clusters=None):
        """
        Perform K-means clustering analysis
        """
        print("\n🎯 K-MEANS CLUSTERING ANALYSIS")
        print("=" * 50)
        
        if n_clusters is None:
            optimal_results = self.find_optimal_clusters(data[available_features])
            n_clusters = optimal_results['optimal_k']
        
        print(f"🔄 Performing K-means with {n_clusters} clusters...")
        
        # Perform K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data[available_features])
        
        # Add cluster labels to data
        data_clustered = data.copy()
        data_clustered['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = data_clustered.groupby('cluster').agg({
            **{feature: ['mean', 'std', 'count'] for feature in available_features},
            'popularity': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
        
        print(f"📊 Cluster Statistics:")
        print(cluster_stats)
        
        # Cluster evaluation
        silhouette_avg = silhouette_score(data[available_features], cluster_labels)
        calinski_score = calinski_harabasz_score(data[available_features], cluster_labels)
        
        print(f"\n📈 Cluster Quality Metrics:")
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        print(f"   Calinski-Harabasz Score: {calinski_score:.1f}")
        
        # Create visualizations
        self._visualize_kmeans_clusters(data_clustered, available_features, n_clusters)
        
        # Store results
        self.cluster_results['kmeans'] = {
            'model': kmeans,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score
        }
        
        return data_clustered, cluster_stats
    
    def _visualize_kmeans_clusters(self, data_clustered, available_features, n_clusters):
        """
        Create visualizations for K-means clustering results
        """
        print("🎨 Creating K-means visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'K-means Clustering Results (k={n_clusters})', fontsize=16, fontweight='bold')
        
        # 1. PCA visualization
        if len(available_features) > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data_clustered[available_features])
            
            scatter = axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                       c=data_clustered['cluster'], cmap='viridis', 
                                       alpha=0.6, s=1)
            axes[0, 0].set_title('PCA Visualization')
            axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
        
        # 2. Cluster sizes
        cluster_sizes = data_clustered['cluster'].value_counts().sort_index()
        axes[0, 1].bar(cluster_sizes.index, cluster_sizes.values, 
                      color='skyblue', alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Number of Tracks')
        
        # Add percentage labels
        total_tracks = len(data_clustered)
        for i, size in enumerate(cluster_sizes.values):
            percentage = (size / total_tracks) * 100
            axes[0, 1].text(i, size + total_tracks*0.01, f'{percentage:.1f}%', 
                           ha='center', va='bottom')
        
        # 3. Feature means by cluster
        feature_means = data_clustered.groupby('cluster')[available_features].mean()
        
        # Create heatmap
        sns.heatmap(feature_means.T, annot=True, cmap='coolwarm', center=0,
                   ax=axes[1, 0], fmt='.3f', cbar_kws={'label': 'Feature Value'})
        axes[1, 0].set_title('Feature Means by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Audio Features')
        
        # 4. Popularity distribution by cluster
        cluster_popularity = [data_clustered[data_clustered['cluster'] == i]['popularity'].values 
                            for i in range(n_clusters)]
        
        bp = axes[1, 1].boxplot(cluster_popularity, patch_artist=True)
        axes[1, 1].set_title('Popularity Distribution by Cluster')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Popularity')
        
        # Color boxplots
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig('kmeans_clustering_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_hierarchical_clustering(self, data, available_features, n_clusters=None):
        """
        Perform hierarchical clustering analysis
        """
        print("\n🌳 HIERARCHICAL CLUSTERING ANALYSIS")
        print("=" * 50)
        
        if n_clusters is None:
            n_clusters = self.cluster_results.get('kmeans', {}).get('n_clusters', 5)
        
        print(f"🔄 Performing hierarchical clustering with {n_clusters} clusters...")
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = hierarchical.fit_predict(data[available_features])
        
        # Add cluster labels to data
        data_clustered = data.copy()
        data_clustered['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = data_clustered.groupby('cluster').agg({
            **{feature: ['mean', 'std', 'count'] for feature in available_features},
            'popularity': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
        
        print(f"📊 Hierarchical Cluster Statistics:")
        print(cluster_stats)
        
        # Create dendrogram
        self._create_dendrogram(data[available_features])
        
        # Cluster evaluation
        silhouette_avg = silhouette_score(data[available_features], cluster_labels)
        calinski_score = calinski_harabasz_score(data[available_features], cluster_labels)
        
        print(f"\n📈 Hierarchical Cluster Quality Metrics:")
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        print(f"   Calinski-Harabasz Score: {calinski_score:.1f}")
        
        # Store results
        self.cluster_results['hierarchical'] = {
            'model': hierarchical,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score
        }
        
        return data_clustered, cluster_stats
    
    def _create_dendrogram(self, data, max_samples=1000):
        """
        Create dendrogram for hierarchical clustering
        """
        print("🌳 Creating dendrogram...")
        
        # Sample data if too large
        if len(data) > max_samples:
            sample_data = data.sample(n=max_samples, random_state=42)
            print(f"   Using {max_samples} samples for dendrogram")
        else:
            sample_data = data
        
        # Calculate linkage matrix
        linkage_matrix = linkage(sample_data, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, truncate_mode='level', p=5, 
                  show_leaf_counts=True, leaf_font_size=10)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
        plt.xlabel('Sample Index or (Cluster Size)')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig('hierarchical_dendrogram.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_dbscan_clustering(self, data, available_features, eps=None, min_samples=None):
        """
        Perform DBSCAN clustering analysis
        """
        print("\n🎯 DBSCAN CLUSTERING ANALYSIS")
        print("=" * 50)
        
        # Auto-tune parameters if not provided
        if eps is None or min_samples is None:
            eps, min_samples = self._tune_dbscan_parameters(data[available_features])
        
        print(f"🔄 Performing DBSCAN with eps={eps:.3f}, min_samples={min_samples}...")
        
        # Perform DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data[available_features])
        
        # Add cluster labels to data
        data_clustered = data.copy()
        data_clustered['cluster'] = cluster_labels
        
        # Analyze results
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"📊 DBSCAN Results:")
        print(f"   Number of clusters: {n_clusters}")
        print(f"   Number of noise points: {n_noise} ({n_noise/len(data)*100:.1f}%)")
        
        if n_clusters > 0:
            # Calculate cluster statistics (excluding noise)
            cluster_data = data_clustered[data_clustered['cluster'] != -1]
            
            if len(cluster_data) > 0:
                cluster_stats = cluster_data.groupby('cluster').agg({
                    **{feature: ['mean', 'std', 'count'] for feature in available_features},
                    'popularity': ['mean', 'std']
                }).round(3)
                
                # Flatten column names
                cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
                
                print(f"📈 Cluster Statistics (excluding noise):")
                print(cluster_stats)
                
                # Cluster evaluation (excluding noise)
                non_noise_data = data[available_features].iloc[cluster_labels != -1]
                non_noise_labels = cluster_labels[cluster_labels != -1]
                
                if len(set(non_noise_labels)) > 1:
                    silhouette_avg = silhouette_score(non_noise_data, non_noise_labels)
                    calinski_score = calinski_harabasz_score(non_noise_data, non_noise_labels)
                    
                    print(f"\n📈 DBSCAN Quality Metrics (excluding noise):")
                    print(f"   Silhouette Score: {silhouette_avg:.3f}")
                    print(f"   Calinski-Harabasz Score: {calinski_score:.1f}")
                else:
                    silhouette_avg = calinski_score = 0
                    print(f"\n⚠️ Cannot calculate quality metrics: insufficient clusters")
            else:
                cluster_stats = pd.DataFrame()
                silhouette_avg = calinski_score = 0
        else:
            cluster_stats = pd.DataFrame()
            silhouette_avg = calinski_score = 0
            print(f"⚠️ No clusters found with current parameters")
        
        # Store results
        self.cluster_results['dbscan'] = {
            'model': dbscan,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette_avg,
            'calinski_score': calinski_score,
            'eps': eps,
            'min_samples': min_samples
        }
        
        return data_clustered, cluster_stats
    
    def _tune_dbscan_parameters(self, data, k=4):
        """
        Auto-tune DBSCAN parameters using k-distance graph
        """
        print("🔧 Auto-tuning DBSCAN parameters...")
        
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors_fit = neighbors.fit(data)
        distances, indices = neighbors_fit.kneighbors(data)
        
        # Sort distances
        distances = np.sort(distances, axis=0)
        distances = distances[:, k-1]  # k-distance
        
        # Find elbow point
        x = np.arange(len(distances))
        eps = self._find_elbow_point(x, distances)
        
        # Convert back to actual distance value
        eps = distances[int(eps)]
        
        # Set min_samples
        min_samples = k
        
        print(f"   Auto-tuned eps: {eps:.3f}")
        print(f"   Auto-tuned min_samples: {min_samples}")
        
        return eps, min_samples
    
    def create_cluster_profiles(self):
        """
        Create detailed profiles for each cluster
        """
        print("\n📊 CREATING CLUSTER PROFILES")
        print("=" * 50)
        
        if 'kmeans' not in self.cluster_results:
            print("❌ No K-means results available")
            return
        
        kmeans_results = self.cluster_results['kmeans']
        cluster_stats = kmeans_results['cluster_stats']
        
        # Create cluster profiles
        profiles = {}
        
        for cluster_id in range(kmeans_results['n_clusters']):
            profile = {
                'cluster_id': cluster_id,
                'size': int(cluster_stats.loc[cluster_id, [col for col in cluster_stats.columns if col.endswith('_count')][0]]),
                'characteristics': {},
                'music_style': '',
                'recommendations': []
            }
            
            # Extract characteristics
            for feature in self.audio_features:
                if f"{feature}_mean" in cluster_stats.columns:
                    mean_val = cluster_stats.loc[cluster_id, f"{feature}_mean"]
                    profile['characteristics'][feature] = mean_val
            
            # Determine music style based on characteristics
            profile['music_style'] = self._determine_music_style(profile['characteristics'])
            
            # Generate recommendations
            profile['recommendations'] = self._generate_cluster_recommendations(profile)
            
            profiles[cluster_id] = profile
        
        # Print profiles
        print("🎵 Cluster Profiles:")
        for cluster_id, profile in profiles.items():
            print(f"\n   Cluster {cluster_id}: {profile['music_style']}")
            print(f"     Size: {profile['size']:,} tracks")
            print(f"     Key characteristics:")
            
            # Show top 3 characteristics
            char_items = list(profile['characteristics'].items())
            char_items.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, value in char_items[:3]:
                level = self._get_feature_level(feature, value)
                print(f"       {feature}: {value:.3f} ({level})")
            
            print(f"     Recommendations: {', '.join(profile['recommendations'])}")
        
        # Save profiles
        import json
        with open('cluster_profiles.json', 'w') as f:
            json.dump(profiles, f, indent=2, default=str)
        
        print(f"\n✅ Cluster profiles saved to 'cluster_profiles.json'")
        
        return profiles
    
    def _determine_music_style(self, characteristics):
        """
        Determine music style based on cluster characteristics
        """
        # Simple rule-based style determination
        energy = characteristics.get('energy', 0)
        danceability = characteristics.get('danceability', 0)
        valence = characteristics.get('valence', 0)
        acousticness = characteristics.get('acousticness', 0)
        
        if energy > 0.7 and danceability > 0.7:
            return "High Energy Dance"
        elif energy > 0.6 and valence > 0.6:
            return "Upbeat & Energetic"
        elif acousticness > 0.6 and valence < 0.4:
            return "Melancholic Acoustic"
        elif acousticness > 0.5:
            return "Acoustic & Chill"
        elif valence > 0.6:
            return "Happy & Positive"
        elif valence < 0.4:
            return "Sad & Emotional"
        elif energy < 0.4:
            return "Calm & Relaxed"
        else:
            return "Balanced Mix"
    
    def _get_feature_level(self, feature, value):
        """
        Get descriptive level for a feature value
        """
        if value > 0.7:
            return "Very High"
        elif value > 0.5:
            return "High"
        elif value > 0.3:
            return "Medium"
        elif value > 0.1:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_cluster_recommendations(self, profile):
        """
        Generate recommendations for a cluster
        """
        recommendations = []
        
        characteristics = profile['characteristics']
        
        # High energy clusters
        if characteristics.get('energy', 0) > 0.7:
            recommendations.append("Great for workout playlists")
            recommendations.append("Perfect for energetic activities")
        
        # High danceability
        if characteristics.get('danceability', 0) > 0.7:
            recommendations.append("Ideal for dance parties")
            recommendations.append("Club and party music")
        
        # High valence
        if characteristics.get('valence', 0) > 0.6:
            recommendations.append("Mood-lifting music")
            recommendations.append("Good for motivation")
        
        # High acousticness
        if characteristics.get('acousticness', 0) > 0.6:
            recommendations.append("Perfect for relaxation")
            recommendations.append("Acoustic sessions")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("General listening")
            recommendations.append("Background music")
        
        return recommendations
    
    def compare_clustering_methods(self):
        """
        Compare different clustering methods
        """
        print("\n🔍 COMPARING CLUSTERING METHODS")
        print("=" * 50)
        
        methods = ['kmeans', 'hierarchical', 'dbscan']
        available_methods = [m for m in methods if m in self.cluster_results]
        
        if len(available_methods) == 0:
            print("❌ No clustering results available for comparison")
            return
        
        print("📊 Clustering Methods Comparison:")
        print(f"{'Method':<15} {'Clusters':<10} {'Silhouette':<12} {'Calinski':<12} {'Special Notes'}")
        print("-" * 65)
        
        for method in available_methods:
            results = self.cluster_results[method]
            n_clusters = results['n_clusters']
            silhouette = results['silhouette_score']
            calinski = results['calinski_score']
            
            special_notes = ""
            if method == 'dbscan':
                special_notes = f"Noise: {results['n_noise']}"
            
            print(f"{method:<15} {n_clusters:<10} {silhouette:<12.3f} {calinski:<12.1f} {special_notes}")
        
        # Determine best method
        best_method = max(available_methods, 
                         key=lambda m: self.cluster_results[m]['silhouette_score'])
        
        print(f"\n🏆 Best performing method: {best_method}")
        print(f"   Silhouette Score: {self.cluster_results[best_method]['silhouette_score']:.3f}")
        
        return best_method
    
    def create_interactive_cluster_visualization(self):
        """
        Create interactive visualization of clustering results
        """
        print("\n🎨 CREATING INTERACTIVE CLUSTER VISUALIZATION")
        print("=" * 50)
        
        if 'kmeans' not in self.cluster_results:
            print("❌ No K-means results available")
            return
        
        # Get clustering data
        scaled_data, available_features = self.prepare_clustering_data()
        if scaled_data is None:
            return
        
        # Add cluster labels
        scaled_data['cluster'] = self.cluster_results['kmeans']['labels']
        
        # Create PCA for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data[available_features])
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add scatter plot for each cluster
        n_clusters = self.cluster_results['kmeans']['n_clusters']
        colors = px.colors.qualitative.Set3[:n_clusters]
        
        for i in range(n_clusters):
            cluster_data = scaled_data[scaled_data['cluster'] == i]
            cluster_pca = pca_result[scaled_data['cluster'] == i]
            
            fig.add_trace(go.Scatter(
                x=cluster_pca[:, 0],
                y=cluster_pca[:, 1],
                mode='markers',
                name=f'Cluster {i}',
                marker=dict(color=colors[i], size=3, opacity=0.7),
                hovertemplate='<b>Cluster %{text}</b><br>' +
                             'PC1: %{x:.3f}<br>' +
                             'PC2: %{y:.3f}<br>' +
                             'Popularity: %{customdata:.1f}<extra></extra>',
                text=[i] * len(cluster_data),
                customdata=cluster_data['popularity']
            ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Cluster Visualization (PCA)',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
            hovermode='closest',
            showlegend=True,
            width=800,
            height=600
        )
        
        # Save as HTML
        fig.write_html("interactive_cluster_visualization.html")
        print("✅ Interactive visualization saved as 'interactive_cluster_visualization.html'")
        
        # Show the plot
        fig.show()
        
        return fig
    
    def generate_clustering_report(self):
        """
        Generate comprehensive clustering analysis report
        """
        print("\n📋 GENERATING CLUSTERING REPORT")
        print("=" * 50)
        
        report = {
            'analysis_summary': {
                'total_tracks_analyzed': 0,
                'audio_features_used': [],
                'methods_applied': list(self.cluster_results.keys()),
                'best_method': '',
                'optimal_clusters': 0
            },
            'method_comparison': {},
            'cluster_profiles': {},
            'recommendations': []
        }
        
        # Fill in analysis summary
        if self.data is not None:
            report['analysis_summary']['total_tracks_analyzed'] = len(self.data)
            report['analysis_summary']['audio_features_used'] = [f for f in self.audio_features if f in self.data.columns]
        
        # Method comparison
        for method, results in self.cluster_results.items():
            report['method_comparison'][method] = {
                'n_clusters': results['n_clusters'],
                'silhouette_score': results['silhouette_score'],
                'calinski_score': results['calinski_score']
            }
            
            if method == 'dbscan':
                report['method_comparison'][method]['n_noise'] = results['n_noise']
        
        # Best method
        if self.cluster_results:
            best_method = max(self.cluster_results.keys(), 
                            key=lambda m: self.cluster_results[m]['silhouette_score'])
            report['analysis_summary']['best_method'] = best_method
            report['analysis_summary']['optimal_clusters'] = self.cluster_results[best_method]['n_clusters']
        
        # Recommendations
        report['recommendations'] = [
            "Use K-means clustering for balanced and interpretable results",
            "Consider hierarchical clustering for understanding music taxonomy",
            "Apply DBSCAN for discovering outlier tracks",
            "Use cluster profiles for targeted music recommendations",
            "Implement multi-level clustering for better granularity"
        ]
        
        # Save report
        import json
        with open('clustering_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Clustering report saved as 'clustering_analysis_report.json'")
        
        return report


def main():
    """
    Main function to execute Phase 3 Clustering Analysis
    """
    print("🎵 SPOTIFY MUSIC RECOMMENDATION SYSTEM - PHASE 3")
    print("=" * 70)
    print("Phase 3: Advanced Clustering Analysis")
    print("=" * 70)
    
    # Initialize clustering analysis
    clustering = SpotifyClustering()
    
    # Load data
    if not clustering.load_data():
        print("❌ Failed to load data")
        return
    
    # Prepare data for clustering
    scaled_data, available_features = clustering.prepare_clustering_data()
    if scaled_data is None:
        print("❌ Failed to prepare clustering data")
        return
    
    results = {}
    
    # 1. K-means clustering
    print("\n" + "="*50)
    print("1. K-MEANS CLUSTERING")
    print("="*50)
    kmeans_data, kmeans_stats = clustering.perform_kmeans_clustering(scaled_data, available_features)
    results['kmeans'] = {'data': kmeans_data, 'stats': kmeans_stats}
    
    # 2. Hierarchical clustering
    print("\n" + "="*50)
    print("2. HIERARCHICAL CLUSTERING")
    print("="*50)
    hierarchical_data, hierarchical_stats = clustering.perform_hierarchical_clustering(scaled_data, available_features)
    results['hierarchical'] = {'data': hierarchical_data, 'stats': hierarchical_stats}
    
    # 3. DBSCAN clustering
    print("\n" + "="*50)
    print("3. DBSCAN CLUSTERING")
    print("="*50)
    dbscan_data, dbscan_stats = clustering.perform_dbscan_clustering(scaled_data, available_features)
    results['dbscan'] = {'data': dbscan_data, 'stats': dbscan_stats}
    
    # 4. Compare methods
    print("\n" + "="*50)
    print("4. CLUSTERING METHODS COMPARISON")
    print("="*50)
    best_method = clustering.compare_clustering_methods()
    results['best_method'] = best_method
    
    # 5. Create cluster profiles
    print("\n" + "="*50)
    print("5. CLUSTER PROFILING")
    print("="*50)
    profiles = clustering.create_cluster_profiles()
    results['profiles'] = profiles
    
    # 6. Interactive visualization
    print("\n" + "="*50)
    print("6. INTERACTIVE VISUALIZATION")
    print("="*50)
    interactive_fig = clustering.create_interactive_cluster_visualization()
    results['interactive_viz'] = interactive_fig
    
    # 7. Generate comprehensive report
    print("\n" + "="*50)
    print("7. COMPREHENSIVE REPORT")
    print("="*50)
    report = clustering.generate_clustering_report()
    results['report'] = report
    
    # Summary
    print("\n✅ Phase 3 Clustering Analysis completed successfully!")
    print("📊 Analysis Results:")
    print(f"   🎯 K-means clusters: {clustering.cluster_results['kmeans']['n_clusters']}")
    print(f"   🌳 Hierarchical clusters: {clustering.cluster_results['hierarchical']['n_clusters']}")
    print(f"   🎯 DBSCAN clusters: {clustering.cluster_results['dbscan']['n_clusters']}")
    print(f"   🏆 Best method: {best_method}")
    print(f"   📋 Cluster profiles created: {len(profiles) if profiles else 0}")
    
    print("\n📁 Files generated:")
    print("   - clustering_evaluation.png")
    print("   - kmeans_clustering_results.png")
    print("   - hierarchical_dendrogram.png")
    print("   - interactive_cluster_visualization.html")
    print("   - cluster_profiles.json")
    print("   - clustering_analysis_report.json")
    
    print("\n🔄 Ready for Phase 4: Recommendation System Development")
    
    return clustering, results


if __name__ == "__main__":
    # Execute Phase 3
    clustering, results = main()
    
    # Additional insights for recommendation system
    print("\n" + "="*50)
    print("🎯 INSIGHTS FOR RECOMMENDATION SYSTEM")
    print("="*50)
    
    if clustering.cluster_results:
        print("💡 Key Findings for Recommendation System:")
        
        # Best clustering method
        best_method = results.get('best_method', 'kmeans')
        best_results = clustering.cluster_results[best_method]
        
        print(f"   🏆 Optimal clustering method: {best_method}")
        print(f"   📊 Optimal number of clusters: {best_results['n_clusters']}")
        print(f"   📈 Quality score: {best_results['silhouette_score']:.3f}")
        
        # Cluster insights
        if 'profiles' in results and results['profiles']:
            print(f"   🎵 Music style clusters identified:")
            for cluster_id, profile in results['profiles'].items():
                print(f"     - Cluster {cluster_id}: {profile['music_style']} ({profile['size']:,} tracks)")
        
        # Recommendations for system design
        print(f"\n   🎯 Recommendations for Phase 4:")
        print(f"     - Use {best_method} clustering for song grouping")
        print(f"     - Implement cluster-based similarity for recommendations")
        print(f"     - Consider within-cluster and cross-cluster recommendations")
        print(f"     - Use cluster profiles for cold-start recommendations")
        print(f"     - Implement multi-level clustering for better granularity")
    
    print("\n🎉 Phase 3 Clustering Analysis Complete!")
    print("🔄 Next: Phase 4 - Build the Recommendation System!")
