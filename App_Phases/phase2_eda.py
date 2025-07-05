"""
Phase 2: Advanced Exploratory Data Analysis
Music Recommendation System - Spotify Dataset

This phase conducts comprehensive exploratory data analysis on the Spotify dataset.
It analyzes trends in sound features over decades, examines popularity and characteristics
of top genres and artists, and generates advanced visualizations and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from wordcloud import WordCloud
import ast
from collections import Counter
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpotifyEDA:
    """
    Advanced Exploratory Data Analysis for Spotify Dataset
    """
    
    def __init__(self, data_path="Djeezy/Spotify_Data"):
        """
        Initialize the EDA class
        
        Args:
            data_path (str): Path to the Spotify data folder
        """
        self.data_path = Path(data_path)
        self.data = None
        self.data_with_genres = None
        self.data_by_year = None
        self.data_by_artist = None
        self.data_by_genres = None
        self.audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                              'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        
    def load_data(self):
        """
        Load all datasets for analysis
        """
        print("🔄 Loading datasets for EDA...")
        
        # Load main dataset
        try:
            self.data = pd.read_csv(self.data_path / "data.csv")
            print(f"✅ Main dataset loaded: {self.data.shape[0]:,} tracks")
        except Exception as e:
            print(f"❌ Error loading main dataset: {e}")
            
        # Load genre dataset
        try:
            self.data_with_genres = pd.read_csv(self.data_path / "data_w_genres.csv")
            print(f"✅ Genre dataset loaded: {self.data_with_genres.shape[0]:,} records")
        except Exception as e:
            print(f"❌ Error loading genre dataset: {e}")
            
        # Load other datasets
        try:
            self.data_by_year = pd.read_csv(self.data_path / "data_by_year.csv")
            self.data_by_artist = pd.read_csv(self.data_path / "data_by_artist.csv")
            self.data_by_genres = pd.read_csv(self.data_path / "data_by_genres.csv")
            print(f"✅ Additional datasets loaded successfully")
        except Exception as e:
            print(f"⚠️ Some additional datasets not available: {e}")
    
    def analyze_temporal_trends(self):
        """
        Analyze trends in sound features over decades
        """
        if self.data is None:
            print("❌ Main dataset not loaded")
            return
            
        print("\n📈 TEMPORAL TRENDS ANALYSIS")
        print("=" * 50)
        
        # Create decade column
        self.data['decade'] = (self.data['year'] // 10) * 10
        
        # Calculate decade-wise statistics
        decade_stats = self.data.groupby('decade').agg({
            **{feature: ['mean', 'std'] for feature in self.audio_features if feature in self.data.columns},
            'popularity': ['mean', 'std', 'count']
        }).round(3)
        
        # Flatten column names
        decade_stats.columns = [f"{col[0]}_{col[1]}" for col in decade_stats.columns]
        
        print("📊 Decade-wise Audio Feature Evolution:")
        print(decade_stats.head(10))
        
        # Create comprehensive temporal visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Evolution of Audio Features Over Decades', fontsize=16, fontweight='bold')
        
        available_features = [f for f in self.audio_features if f in self.data.columns]
        
        for i, feature in enumerate(available_features[:9]):
            row, col = i // 3, i % 3
            
            # Calculate decade averages
            decade_avg = self.data.groupby('decade')[feature].mean()
            
            # Plot trend line
            axes[row, col].plot(decade_avg.index, decade_avg.values, 
                               marker='o', linewidth=2, markersize=6)
            axes[row, col].set_title(f'{feature.capitalize()} Evolution')
            axes[row, col].set_xlabel('Decade')
            axes[row, col].set_ylabel(f'Average {feature.capitalize()}')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add trend annotation
            slope, intercept, r_value, p_value, std_err = stats.linregress(decade_avg.index, decade_avg.values)
            trend = "↗️ Increasing" if slope > 0 else "↘️ Decreasing"
            axes[row, col].text(0.02, 0.98, f'{trend} (r²={r_value**2:.3f})', 
                               transform=axes[row, col].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('temporal_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical significance testing
        print("\n🔬 Statistical Significance Tests:")
        print("Testing if audio features have changed significantly over decades...")
        
        decades = sorted(self.data['decade'].unique())
        if len(decades) >= 2:
            early_decade = decades[0]
            late_decade = decades[-1]
            
            for feature in available_features:
                early_data = self.data[self.data['decade'] == early_decade][feature].dropna()
                late_data = self.data[self.data['decade'] == late_decade][feature].dropna()
                
                if len(early_data) > 0 and len(late_data) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(early_data, late_data)
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    change = "Increased" if late_data.mean() > early_data.mean() else "Decreased"
                    
                    print(f"   {feature}: {change} from {early_decade}s to {late_decade}s "
                          f"({significance}, p={p_value:.4f})")
        
        return decade_stats
    
    def analyze_popularity_characteristics(self):
        """
        Examine popularity and characteristics of top genres and artists
        """
        print("\n⭐ POPULARITY ANALYSIS")
        print("=" * 50)
        
        if self.data is None:
            print("❌ Main dataset not loaded")
            return
            
        # Artist popularity analysis
        print("🎤 TOP ARTISTS ANALYSIS:")
        artist_stats = self.data.groupby('artists').agg({
            'popularity': ['mean', 'std', 'count'],
            'energy': 'mean',
            'danceability': 'mean',
            'valence': 'mean'
        }).round(3)
        
        # Flatten columns
        artist_stats.columns = [f"{col[0]}_{col[1]}" for col in artist_stats.columns]
        
        # Filter artists with at least 5 tracks
        top_artists = artist_stats[artist_stats['popularity_count'] >= 5].sort_values(
            'popularity_mean', ascending=False
        ).head(20)
        
        print("Top 20 Most Popular Artists (min 5 tracks):")
        for artist, stats in top_artists.iterrows():
            print(f"   {artist}: {stats['popularity_mean']:.1f} avg popularity "
                  f"({stats['popularity_count']:.0f} tracks)")
        
        # Popularity distribution analysis
        print("\n📊 POPULARITY DISTRIBUTION:")
        
        # Create popularity bins
        self.data['popularity_tier'] = pd.cut(self.data['popularity'], 
                                             bins=[0, 25, 50, 75, 100],
                                             labels=['Low', 'Medium', 'High', 'Very High'])
        
        popularity_distribution = self.data['popularity_tier'].value_counts()
        print("Distribution by popularity tier:")
        for tier, count in popularity_distribution.items():
            percentage = (count / len(self.data)) * 100
            print(f"   {tier}: {count:,} tracks ({percentage:.1f}%)")
        
        # Audio features by popularity tier
        print("\n🎵 AUDIO FEATURES BY POPULARITY TIER:")
        available_features = [f for f in self.audio_features if f in self.data.columns]
        
        tier_features = self.data.groupby('popularity_tier')[available_features].mean()
        
        for tier in ['Low', 'Medium', 'High', 'Very High']:
            if tier in tier_features.index:
                print(f"\n   {tier} Popularity Tracks:")
                for feature in available_features:
                    print(f"     {feature}: {tier_features.loc[tier, feature]:.3f}")
        
        # Create popularity analysis visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Popularity Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Popularity distribution
        axes[0, 0].hist(self.data['popularity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Popularity Distribution')
        axes[0, 0].set_xlabel('Popularity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.data['popularity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.data["popularity"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Top artists bar chart
        top_10_artists = top_artists.head(10)
        axes[0, 1].barh(range(len(top_10_artists)), top_10_artists['popularity_mean'], 
                       color='orange', alpha=0.8)
        axes[0, 1].set_yticks(range(len(top_10_artists)))
        axes[0, 1].set_yticklabels([artist[:20] + '...' if len(artist) > 20 else artist 
                                   for artist in top_10_artists.index])
        axes[0, 1].set_title('Top 10 Artists by Average Popularity')
        axes[0, 1].set_xlabel('Average Popularity')
        
        # 3. Popularity tier distribution
        tier_counts = self.data['popularity_tier'].value_counts()
        axes[1, 0].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
                      colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold'])
        axes[1, 0].set_title('Popularity Tier Distribution')
        
        # 4. Audio features heatmap by popularity tier
        if len(available_features) > 0:
            tier_features_normalized = tier_features.T
            sns.heatmap(tier_features_normalized, annot=True, cmap='viridis', 
                       ax=axes[1, 1], fmt='.3f', cbar_kws={'label': 'Feature Value'})
            axes[1, 1].set_title('Audio Features by Popularity Tier')
            axes[1, 1].set_xlabel('Popularity Tier')
            axes[1, 1].set_ylabel('Audio Features')
        
        plt.tight_layout()
        plt.savefig('popularity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'top_artists': top_artists,
            'popularity_distribution': popularity_distribution,
            'tier_features': tier_features
        }
    
    def create_genre_analysis(self):
        """
        Analyze genres and create word clouds
        """
        print("\n🎼 GENRE ANALYSIS")
        print("=" * 50)
        
        if self.data_with_genres is None:
            print("❌ Genre dataset not loaded")
            return
            
        # Extract all genres
        all_genres = []
        genre_track_count = {}
        
        print("📋 Processing genre data...")
        
        for idx, row in self.data_with_genres.iterrows():
            if pd.notna(row['genres']):
                try:
                    # Parse the genre list (assuming it's stored as string representation of list)
                    if isinstance(row['genres'], str):
                        # Remove brackets and quotes, split by comma
                        genres = row['genres'].strip('[]').replace("'", "").replace('"', '')
                        genres = [g.strip() for g in genres.split(',') if g.strip()]
                    else:
                        genres = row['genres'] if isinstance(row['genres'], list) else []
                    
                    all_genres.extend(genres)
                    
                    # Count tracks per genre
                    for genre in genres:
                        genre_track_count[genre] = genre_track_count.get(genre, 0) + 1
                        
                except Exception as e:
                    continue
        
        # Genre statistics
        genre_counter = Counter(all_genres)
        most_common_genres = genre_counter.most_common(20)
        
        print(f"📊 Genre Statistics:")
        print(f"   Total unique genres: {len(genre_counter)}")
        print(f"   Most common genres:")
        
        for genre, count in most_common_genres:
            percentage = (count / len(all_genres)) * 100
            print(f"     {genre}: {count:,} ({percentage:.1f}%)")
        
        # Create word cloud
        if len(all_genres) > 0:
            print("\n🎨 Creating genre word cloud...")
            
            # Prepare text for word cloud
            genre_text = ' '.join(all_genres)
            
            # Create word cloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                colormap='viridis',
                                max_words=100,
                                relative_scaling=0.5).generate(genre_text)
            
            # Display word cloud
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Spotify Genres Word Cloud', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('genre_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Genre network analysis (if enough data)
        if len(most_common_genres) >= 5:
            print("\n🕸️ Creating genre network analysis...")
            self.create_genre_network(most_common_genres)
        
        # Create genre visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Genre Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Top genres bar chart
        top_genres = dict(most_common_genres[:15])
        axes[0, 0].barh(range(len(top_genres)), list(top_genres.values()), 
                       color='lightgreen', alpha=0.8)
        axes[0, 0].set_yticks(range(len(top_genres)))
        axes[0, 0].set_yticklabels([genre[:20] + '...' if len(genre) > 20 else genre 
                                   for genre in top_genres.keys()])
        axes[0, 0].set_title('Top 15 Most Common Genres')
        axes[0, 0].set_xlabel('Frequency')
        
        # 2. Genre distribution (pie chart for top 10)
        top_10_genres = dict(most_common_genres[:10])
        other_count = sum(dict(most_common_genres[10:]).values())
        if other_count > 0:
            top_10_genres['Others'] = other_count
            
        axes[0, 1].pie(top_10_genres.values(), labels=top_10_genres.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Top 10 Genres Distribution')
        
        # 3. Genre frequency distribution
        genre_counts = list(genre_counter.values())
        axes[1, 0].hist(genre_counts, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Genre Frequency Distribution')
        axes[1, 0].set_xlabel('Number of Tracks')
        axes[1, 0].set_ylabel('Number of Genres')
        axes[1, 0].set_yscale('log')
        
        # 4. Genre diversity over time (if year data available)
        if 'year' in self.data_with_genres.columns:
            yearly_genre_diversity = []
            years = sorted(self.data_with_genres['year'].unique())
            
            for year in years:
                year_data = self.data_with_genres[self.data_with_genres['year'] == year]
                year_genres = []
                for genres in year_data['genres'].dropna():
                    if isinstance(genres, str):
                        year_genres.extend(genres.strip('[]').replace("'", "").split(', '))
                
                unique_genres = len(set(year_genres))
                yearly_genre_diversity.append(unique_genres)
            
            axes[1, 1].plot(years, yearly_genre_diversity, marker='o', linewidth=2)
            axes[1, 1].set_title('Genre Diversity Over Time')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Number of Unique Genres')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'genre_counter': genre_counter,
            'most_common_genres': most_common_genres,
            'total_genres': len(genre_counter)
        }
    
    def create_genre_network(self, top_genres):
        """
        Create a network analysis of genre co-occurrences
        """
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (genres)
            for genre, count in top_genres[:10]:  # Top 10 genres
                G.add_node(genre, weight=count)
            
            # Add edges based on co-occurrence
            genre_cooccurrence = {}
            
            for idx, row in self.data_with_genres.iterrows():
                if pd.notna(row['genres']):
                    try:
                        if isinstance(row['genres'], str):
                            genres = row['genres'].strip('[]').replace("'", "").replace('"', '')
                            genres = [g.strip() for g in genres.split(',') if g.strip()]
                        else:
                            genres = row['genres'] if isinstance(row['genres'], list) else []
                        
                        # Find genre pairs
                        top_genre_names = [g[0] for g in top_genres[:10]]
                        track_genres = [g for g in genres if g in top_genre_names]
                        
                        for i in range(len(track_genres)):
                            for j in range(i+1, len(track_genres)):
                                pair = tuple(sorted([track_genres[i], track_genres[j]]))
                                genre_cooccurrence[pair] = genre_cooccurrence.get(pair, 0) + 1
                    except:
                        continue
            
            # Add edges with weights
            for (genre1, genre2), weight in genre_cooccurrence.items():
                if weight > 5:  # Only show significant co-occurrences
                    G.add_edge(genre1, genre2, weight=weight)
            
            # Create network visualization
            plt.figure(figsize=(12, 8))
            
            # Position nodes
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw nodes
            node_sizes = [G.nodes[node]['weight'] * 10 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color='lightblue', alpha=0.8)
            
            # Draw edges
            edge_widths = [G.edges[edge]['weight'] / 10 for edge in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            plt.title('Genre Co-occurrence Network', fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('genre_network.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Genre network analysis completed")
            
        except Exception as e:
            print(f"⚠️ Network analysis skipped: {e}")
    
    def perform_advanced_statistical_analysis(self):
        """
        Perform advanced statistical analysis including PCA and correlation analysis
        """
        print("\n🔬 ADVANCED STATISTICAL ANALYSIS")
        print("=" * 50)
        
        if self.data is None:
            print("❌ Main dataset not loaded")
            return
            
        # Select numerical features for analysis
        available_features = [f for f in self.audio_features if f in self.data.columns]
        if len(available_features) < 3:
            print("❌ Insufficient numerical features for analysis")
            return
            
        # Prepare data for PCA
        feature_data = self.data[available_features + ['popularity']].dropna()
        
        # Correlation Analysis
        print("🔗 CORRELATION ANALYSIS:")
        correlation_matrix = feature_data.corr()
        
        # Find strongest correlations
        print("Top correlations with popularity:")
        pop_correlations = correlation_matrix['popularity'].drop('popularity').abs().sort_values(ascending=False)
        for feature, corr in pop_correlations.head(5).items():
            print(f"   {feature}: {corr:.3f}")
        
        # Principal Component Analysis
        print("\n🧮 PRINCIPAL COMPONENT ANALYSIS:")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_data[available_features])
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(features_scaled)
        
        # PCA results
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"Number of components: {len(explained_variance_ratio)}")
        print("Explained variance ratio by component:")
        for i, ratio in enumerate(explained_variance_ratio):
            print(f"   PC{i+1}: {ratio:.3f} ({cumulative_variance[i]:.3f} cumulative)")
        
        # Create PCA visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Statistical Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   mask=mask, ax=axes[0, 0], fmt='.2f')
        axes[0, 0].set_title('Feature Correlation Matrix')
        
        # 2. PCA explained variance
        axes[0, 1].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        axes[0, 1].plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, 
                       'ro-', alpha=0.7)
        axes[0, 1].set_title('PCA Explained Variance')
        axes[0, 1].set_xlabel('Principal Component')
        axes[0, 1].set_ylabel('Explained Variance Ratio')
        axes[0, 1].legend(['Individual', 'Cumulative'])
        
        # 3. PCA scatter plot (first two components)
        if len(pca_result) > 0:
            scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                       c=feature_data['popularity'], cmap='viridis', 
                                       alpha=0.6, s=1)
            axes[1, 0].set_title('PCA Scatter Plot (PC1 vs PC2)')
            axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
            plt.colorbar(scatter, ax=axes[1, 0], label='Popularity')
        
        # 4. Feature importance in first two PCs
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'PC1': pca.components_[0],
            'PC2': pca.components_[1] if len(pca.components_) > 1 else [0] * len(available_features)
        })
        
        x_pos = np.arange(len(available_features))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, feature_importance['PC1'], width, 
                      label='PC1', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, feature_importance['PC2'], width, 
                      label='PC2', alpha=0.8)
        axes[1, 1].set_title('Feature Loadings in Principal Components')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Loading')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(available_features, rotation=45, ha='right')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('advanced_statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'correlation_matrix': correlation_matrix,
            'pca_explained_variance': explained_variance_ratio,
            'pca_components': pca.components_,
            'feature_importance': feature_importance
        }
    
    def create_interactive_dashboard(self):
        """
        Create an interactive dashboard using Plotly
        """
        print("\n🎨 CREATING INTERACTIVE DASHBOARD")
        print("=" * 50)
        
        if self.data is None:
            print("❌ Main dataset not loaded")
            return
            
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Popularity Distribution', 'Audio Features Over Time', 
                           'Feature Correlation Network', 'Genre Analysis'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Popularity distribution
        fig.add_trace(
            go.Histogram(x=self.data['popularity'], nbinsx=50, name='Popularity'),
            row=1, col=1
        )
        
        # 2. Audio features over time
        if 'year' in self.data.columns:
            available_features = [f for f in self.audio_features[:3] if f in self.data.columns]
            colors = ['red', 'blue', 'green']
            
            for i, feature in enumerate(available_features):
                yearly_avg = self.data.groupby('year')[feature].mean()
                fig.add_trace(
                    go.Scatter(x=yearly_avg.index, y=yearly_avg.values, 
                              mode='lines+markers', name=feature, 
                              line=dict(color=colors[i % len(colors)])),
                    row=1, col=2
                )
        
        # 3. Feature correlation (scatter plot)
        if 'energy' in self.data.columns and 'danceability' in self.data.columns:
            fig.add_trace(
                go.Scatter(x=self.data['energy'], y=self.data['danceability'],
                          mode='markers', name='Energy vs Danceability',
                          marker=dict(color=self.data['popularity'], 
                                    colorscale='viridis', showscale=True),
                          text=self.data['name'] if 'name' in self.data.columns else None),
                row=2, col=1
            )
        
        # 4. Top artists
        if 'artists' in self.data.columns:
            top_artists = self.data['artists'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=top_artists.values, y=top_artists.index, 
                      orientation='h', name='Top Artists'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Spotify Dataset Interactive Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save as HTML
        fig.write_html("spotify_interactive_dashboard.html")
        print("✅ Interactive dashboard saved as 'spotify_interactive_dashboard.html'")
        
        # Show the plot
        fig.show()
        
        return fig
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive EDA report
        """
        print("\n📊 GENERATING COMPREHENSIVE EDA REPORT")
        print("=" * 50)
        
        report = {
            'dataset_overview': {},
            'temporal_analysis': {},
            'popularity_analysis': {},
            'genre_analysis': {},
            'statistical_analysis': {},
            'recommendations': []
        }
        
        if self.data is not None:
            # Dataset overview
            report['dataset_overview'] = {
                'total_tracks': len(self.data),
                'unique_artists': len(self.data['artists'].unique()) if 'artists' in self.data.columns else 0,
                'year_range': (self.data['year'].min(), self.data['year'].max()) if 'year' in self.data.columns else None,
                'audio_features_available': len([f for f in self.audio_features if f in self.data.columns])
            }
            
            # Key findings
            available_features = [f for f in self.audio_features if f in self.data.columns]
            if available_features:
                # Most/least energetic, danceable, etc.
                insights = {}
                for feature in available_features:
                    insights[f'highest_{feature}'] = {
                        'value': self.data[feature].max(),
                        'track': self.data.loc[self.data[feature].idxmax(), 'name'] if 'name' in self.data.columns else 'Unknown'
                    }
                    insights[f'lowest_{feature}'] = {
                        'value': self.data[feature].min(),
                        'track': self.data.loc[self.data[feature].idxmin(), 'name'] if 'name' in self.data.columns else 'Unknown'
                    }
                
                report['key_insights'] = insights
            
            # Recommendations for recommendation system
            report['recommendations'] = [
                "Use energy and danceability as primary features for recommendation",
                "Consider temporal trends when recommending music",
                "Implement genre-based filtering for better recommendations",
                "Use popularity as a secondary ranking factor",
                "Consider audio feature correlations for diverse recommendations"
            ]
        
        # Save report
        import json
        with open('eda_comprehensive_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Comprehensive EDA report saved as 'eda_comprehensive_report.json'")
        
        return report


def main():
    """
    Main function to execute Phase 2 EDA
    """
    print("🎵 SPOTIFY MUSIC RECOMMENDATION SYSTEM - PHASE 2")
    print("=" * 70)
    print("Phase 2: Advanced Exploratory Data Analysis")
    print("=" * 70)
    
    # Initialize EDA
    eda = SpotifyEDA()
    
    # Load data
    eda.load_data()
    
    results = {}
    
    # 1. Temporal trends analysis
    print("\n" + "="*50)
    print("1. TEMPORAL TRENDS ANALYSIS")
    print("="*50)
    temporal_results = eda.analyze_temporal_trends()
    results['temporal_analysis'] = temporal_results
    
    # 2. Popularity analysis
    print("\n" + "="*50)
    print("2. POPULARITY ANALYSIS")
    print("="*50)
    popularity_results = eda.analyze_popularity_characteristics()
    results['popularity_analysis'] = popularity_results
    
    # 3. Genre analysis
    print("\n" + "="*50)
    print("3. GENRE ANALYSIS")
    print("="*50)
    genre_results = eda.create_genre_analysis()
    results['genre_analysis'] = genre_results
    
    # 4. Advanced statistical analysis
    print("\n" + "="*50)
    print("4. ADVANCED STATISTICAL ANALYSIS")
    print("="*50)
    statistical_results = eda.perform_advanced_statistical_analysis()
    results['statistical_analysis'] = statistical_results
    
    # 5. Interactive dashboard
    print("\n" + "="*50)
    print("5. INTERACTIVE DASHBOARD")
    print("="*50)
    dashboard = eda.create_interactive_dashboard()
    results['dashboard'] = dashboard
    
    # 6. Comprehensive report
    print("\n" + "="*50)
    print("6. COMPREHENSIVE REPORT")
    print("="*50)
    report = eda.generate_comprehensive_report()
    results['report'] = report
    
    # Summary
    print("\n✅ Phase 2 EDA completed successfully!")
    print("📊 Analysis Results:")
    print(f"   📈 Temporal trends analyzed across decades")
    print(f"   ⭐ Popularity characteristics identified")
    print(f"   🎼 Genre analysis with {results['genre_analysis']['total_genres'] if 'genre_analysis' in results else 0} unique genres")
    print(f"   🔬 Advanced statistical analysis performed")
    print(f"   🎨 Interactive dashboard created")
    print(f"   📋 Comprehensive report generated")
    
    print("\n📁 Files generated:")
    print("   - temporal_trends_analysis.png")
    print("   - popularity_analysis.png")
    print("   - genre_analysis.png")
    print("   - genre_wordcloud.png")
    print("   - genre_network.png")
    print("   - advanced_statistical_analysis.png")
    print("   - spotify_interactive_dashboard.html")
    print("   - eda_comprehensive_report.json")
    
    print("\n🔄 Ready for Phase 3: Clustering Analysis")
    
    return eda, results


if __name__ == "__main__":
    # Execute Phase 2
    eda, results = main()
    
    # Additional insights
    print("\n" + "="*50)
    print("🎯 KEY INSIGHTS FOR RECOMMENDATION SYSTEM")
    print("="*50)
    
    if eda.data is not None:
        print("💡 Recommendation System Insights:")
        
        # Audio feature insights
        available_features = [f for f in eda.audio_features if f in eda.data.columns]
        if available_features and 'popularity' in eda.data.columns:
            feature_popularity_corr = eda.data[available_features + ['popularity']].corr()['popularity'].drop('popularity')
            top_features = feature_popularity_corr.abs().sort_values(ascending=False).head(3)
            
            print(f"   🎵 Top 3 features correlated with popularity:")
            for feature, corr in top_features.items():
                print(f"     - {feature}: {corr:.3f}")
        
        # Year insights
        if 'year' in eda.data.columns:
            decade_popularity = eda.data.groupby((eda.data['year'] // 10) * 10)['popularity'].mean()
            best_decade = decade_popularity.idxmax()
            print(f"   📅 Most popular decade: {best_decade}s (avg popularity: {decade_popularity.max():.1f})")
        
        # Artist insights
        if 'artists' in eda.data.columns:
            prolific_artists = eda.data['artists'].value_counts().head(5)
            print(f"   🎤 Most prolific artists:")
            for artist, count in prolific_artists.items():
                print(f"     - {artist}: {count} tracks")
    
    print("\n🎉 Phase 2 Advanced EDA Complete!")
    print("🔄 Next: Phase 3 - Clustering Analysis for music grouping")
