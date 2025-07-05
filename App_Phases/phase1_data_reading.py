"""
Phase 1: Reading Data
Music Recommendation System - Spotify Dataset

This phase focuses on reading the Spotify dataset and exploring its contents.
The objective is to understand the data structure and prepare it for subsequent phases.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpotifyDataReader:
    """
    A class to handle reading and initial exploration of Spotify dataset
    """
    
    def __init__(self, data_path="Djeezy/Spotify_Data"):
        """
        Initialize the data reader with the path to Spotify data
        
        Args:
            data_path (str): Path to the Spotify data folder
        """
        self.data_path = Path(data_path)
        self.data = None
        self.data_with_genres = None
        self.data_by_year = None
        self.data_by_artist = None
        self.data_by_genres = None
        
    def load_main_dataset(self):
        """
        Load the main Spotify dataset
        
        Returns:
            pd.DataFrame: Main Spotify dataset
        """
        try:
            file_path = self.data_path / "data.csv"
            self.data = pd.read_csv(file_path)
            print(f"✅ Successfully loaded main dataset: {self.data.shape[0]} tracks")
            return self.data
        except FileNotFoundError:
            print(f"❌ Main dataset not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading main dataset: {e}")
            return None
    
    def load_genre_dataset(self):
        """
        Load the dataset with genre information
        
        Returns:
            pd.DataFrame: Dataset with genre information
        """
        try:
            file_path = self.data_path / "data_w_genres.csv"
            self.data_with_genres = pd.read_csv(file_path)
            print(f"✅ Successfully loaded genre dataset: {self.data_with_genres.shape[0]} records")
            return self.data_with_genres
        except FileNotFoundError:
            print(f"❌ Genre dataset not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading genre dataset: {e}")
            return None
    
    def load_year_dataset(self):
        """
        Load the dataset aggregated by year
        
        Returns:
            pd.DataFrame: Dataset aggregated by year
        """
        try:
            file_path = self.data_path / "data_by_year.csv"
            self.data_by_year = pd.read_csv(file_path)
            print(f"✅ Successfully loaded year dataset: {self.data_by_year.shape[0]} years")
            return self.data_by_year
        except FileNotFoundError:
            print(f"❌ Year dataset not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading year dataset: {e}")
            return None
    
    def load_artist_dataset(self):
        """
        Load the dataset aggregated by artist
        
        Returns:
            pd.DataFrame: Dataset aggregated by artist
        """
        try:
            file_path = self.data_path / "data_by_artist.csv"
            self.data_by_artist = pd.read_csv(file_path)
            print(f"✅ Successfully loaded artist dataset: {self.data_by_artist.shape[0]} artists")
            return self.data_by_artist
        except FileNotFoundError:
            print(f"❌ Artist dataset not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading artist dataset: {e}")
            return None
    
    def load_genres_dataset(self):
        """
        Load the dataset aggregated by genres
        
        Returns:
            pd.DataFrame: Dataset aggregated by genres
        """
        try:
            file_path = self.data_path / "data_by_genres.csv"
            self.data_by_genres = pd.read_csv(file_path)
            print(f"✅ Successfully loaded genres dataset: {self.data_by_genres.shape[0]} genre entries")
            return self.data_by_genres
        except FileNotFoundError:
            print(f"❌ Genres dataset not found at {file_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading genres dataset: {e}")
            return None
    
    def load_all_datasets(self):
        """
        Load all available datasets
        
        Returns:
            dict: Dictionary containing all loaded datasets
        """
        print("🔄 Loading all Spotify datasets...")
        print("=" * 50)
        
        datasets = {
            'main': self.load_main_dataset(),
            'genres': self.load_genre_dataset(),
            'by_year': self.load_year_dataset(),
            'by_artist': self.load_artist_dataset(),
            'by_genres': self.load_genres_dataset()
        }
        
        print("=" * 50)
        print("📊 Dataset Loading Summary:")
        for name, dataset in datasets.items():
            if dataset is not None:
                print(f"   {name}: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
            else:
                print(f"   {name}: Failed to load")
        
        return datasets
    
    def explore_main_dataset(self):
        """
        Explore the main dataset structure and basic statistics
        """
        if self.data is None:
            print("❌ Main dataset not loaded. Please load it first.")
            return
        
        print("\n🔍 MAIN DATASET EXPLORATION")
        print("=" * 50)
        
        # Basic info
        print(f"📈 Dataset Shape: {self.data.shape}")
        print(f"📅 Year Range: {self.data['year'].min()} - {self.data['year'].max()}")
        print(f"🎵 Total Tracks: {self.data.shape[0]:,}")
        print(f"🎤 Unique Artists: {len(self.data['artists'].unique()):,}")
        
        # Column information
        print("\n📋 Column Information:")
        print(self.data.dtypes)
        
        # Missing values
        print("\n🔍 Missing Values:")
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Basic statistics for audio features
        print("\n📊 Audio Features Statistics:")
        audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        
        for feature in audio_features:
            if feature in self.data.columns:
                print(f"   {feature}: {self.data[feature].mean():.3f} (±{self.data[feature].std():.3f})")
        
        # Popularity distribution
        print(f"\n⭐ Popularity Statistics:")
        print(f"   Mean: {self.data['popularity'].mean():.2f}")
        print(f"   Median: {self.data['popularity'].median():.2f}")
        print(f"   Most Popular Track: {self.data.loc[self.data['popularity'].idxmax(), 'name']}")
        
        # Display sample data
        print("\n📋 Sample Data (First 5 rows):")
        print(self.data.head())
        
        return self.data.describe()
    
    def explore_genre_dataset(self):
        """
        Explore the genre dataset structure
        """
        if self.data_with_genres is None:
            print("❌ Genre dataset not loaded. Please load it first.")
            return
        
        print("\n🔍 GENRE DATASET EXPLORATION")
        print("=" * 50)
        
        print(f"📈 Dataset Shape: {self.data_with_genres.shape}")
        print(f"🎵 Total Records: {self.data_with_genres.shape[0]:,}")
        
        # Column information
        print("\n📋 Columns:", list(self.data_with_genres.columns))
        
        # Sample data
        print("\n📋 Sample Data (First 5 rows):")
        print(self.data_with_genres.head())
        
        return self.data_with_genres.describe()
    
    def create_data_summary_report(self):
        """
        Create a comprehensive summary report of all datasets
        """
        print("\n📊 COMPREHENSIVE DATA SUMMARY REPORT")
        print("=" * 60)
        
        # Load all datasets if not already loaded
        datasets = self.load_all_datasets()
        
        summary_report = {
            'datasets_loaded': len([d for d in datasets.values() if d is not None]),
            'total_tracks': 0,
            'year_range': None,
            'audio_features': [],
            'datasets_info': {}
        }
        
        # Main dataset analysis
        if self.data is not None:
            summary_report['total_tracks'] = self.data.shape[0]
            summary_report['year_range'] = (self.data['year'].min(), self.data['year'].max())
            
            # Audio features
            audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                            'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
            summary_report['audio_features'] = [f for f in audio_features if f in self.data.columns]
        
        # Store info for each dataset
        for name, dataset in datasets.items():
            if dataset is not None:
                summary_report['datasets_info'][name] = {
                    'shape': dataset.shape,
                    'columns': list(dataset.columns),
                    'memory_usage': dataset.memory_usage(deep=True).sum()
                }
        
        # Print summary
        print(f"✅ Successfully loaded {summary_report['datasets_loaded']}/5 datasets")
        print(f"🎵 Total tracks in main dataset: {summary_report['total_tracks']:,}")
        if summary_report['year_range']:
            print(f"📅 Year range: {summary_report['year_range'][0]} - {summary_report['year_range'][1]}")
        print(f"🎼 Audio features available: {len(summary_report['audio_features'])}")
        
        return summary_report
    
    def save_exploration_results(self, output_path="data_exploration_results.txt"):
        """
        Save exploration results to a text file
        
        Args:
            output_path (str): Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                f.write("SPOTIFY DATA EXPLORATION RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                if self.data is not None:
                    f.write(f"Main Dataset Shape: {self.data.shape}\n")
                    f.write(f"Year Range: {self.data['year'].min()} - {self.data['year'].max()}\n")
                    f.write(f"Total Tracks: {self.data.shape[0]:,}\n")
                    f.write(f"Unique Artists: {len(self.data['artists'].unique()):,}\n\n")
                    
                    f.write("Audio Features Statistics:\n")
                    audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                                    'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
                    for feature in audio_features:
                        if feature in self.data.columns:
                            f.write(f"   {feature}: {self.data[feature].mean():.3f} (±{self.data[feature].std():.3f})\n")
            
            print(f"✅ Exploration results saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def export_to_multiple_formats(self, base_name="spotify_data_processed"):
        """
        Export processed data to multiple formats
        Demonstrates file handling skills
        """
        if self.data is None:
            print("❌ Main dataset not loaded. Please load it first.")
            return
            
        print("\n💾 EXPORTING DATA TO MULTIPLE FORMATS")
        print("=" * 50)
        
        try:
            # Export to CSV
            csv_path = f"{base_name}.csv"
            self.data.to_csv(csv_path, index=False)
            print(f"✅ CSV exported: {csv_path}")
            
            # Export to JSON
            json_path = f"{base_name}.json"
            self.data.to_json(json_path, orient='records', indent=2)
            print(f"✅ JSON exported: {json_path}")
            
            # Export to Excel (if openpyxl is available)
            try:
                excel_path = f"{base_name}.xlsx"
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    self.data.to_excel(writer, sheet_name='Main_Data', index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': ['Total Tracks', 'Unique Artists', 'Year Range', 'Avg Popularity'],
                        'Value': [
                            len(self.data),
                            len(self.data['artists'].unique()) if 'artists' in self.data.columns else 'N/A',
                            f"{self.data['year'].min()}-{self.data['year'].max()}" if 'year' in self.data.columns else 'N/A',
                            f"{self.data['popularity'].mean():.2f}" if 'popularity' in self.data.columns else 'N/A'
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                print(f"✅ Excel exported: {excel_path}")
            except ImportError:
                print("⚠️ Excel export skipped (openpyxl not available)")
            
            # Export sample data as Parquet (if pyarrow is available)
            try:
                parquet_path = f"{base_name}.parquet"
                self.data.to_parquet(parquet_path, index=False)
                print(f"✅ Parquet exported: {parquet_path}")
            except ImportError:
                print("⚠️ Parquet export skipped (pyarrow not available)")
            
        except Exception as e:
            print(f"❌ Error during export: {e}")
    
    def create_data_dictionary(self):
        """
        Create a comprehensive data dictionary
        """
        print("\n📚 CREATING DATA DICTIONARY")
        print("=" * 50)
        
        data_dictionary = {
            'dataset_info': {
                'name': 'Spotify Music Dataset',
                'description': 'Comprehensive dataset of music tracks with audio features',
                'source': 'Spotify API',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'columns': {}
        }
        
        # Column descriptions
        column_descriptions = {
            'valence': 'Musical positivity (0.0-1.0)',
            'acousticness': 'Acoustic quality measure (0.0-1.0)',
            'danceability': 'How suitable for dancing (0.0-1.0)',
            'energy': 'Perceptual measure of intensity (0.0-1.0)',
            'instrumentalness': 'Predicts whether track contains no vocals (0.0-1.0)',
            'liveness': 'Detects presence of audience (0.0-1.0)',
            'loudness': 'Overall loudness in decibels (dB)',
            'speechiness': 'Presence of spoken words (0.0-1.0)',
            'tempo': 'Beats per minute (BPM)',
            'popularity': 'Track popularity score (0-100)',
            'year': 'Release year',
            'artists': 'Artist name(s)',
            'name': 'Track name',
            'id': 'Spotify track ID'
        }
        
        if self.data is not None:
            for column in self.data.columns:
                col_info = {
                    'data_type': str(self.data[column].dtype),
                    'description': column_descriptions.get(column, 'No description available'),
                    'non_null_count': int(self.data[column].count()),
                    'null_count': int(self.data[column].isnull().sum()),
                    'unique_values': int(self.data[column].nunique())
                }
                
                if self.data[column].dtype in ['int64', 'float64']:
                    col_info.update({
                        'min_value': float(self.data[column].min()),
                        'max_value': float(self.data[column].max()),
                        'mean_value': float(self.data[column].mean()),
                        'std_value': float(self.data[column].std())
                    })
                elif self.data[column].dtype == 'object':
                    col_info.update({
                        'sample_values': self.data[column].dropna().head(3).tolist()
                    })
                
                data_dictionary['columns'][column] = col_info
        
        # Save data dictionary
        dict_path = "data_dictionary.json"
        with open(dict_path, 'w') as f:
            json.dump(data_dictionary, f, indent=2, default=str)
        
        print(f"✅ Data dictionary saved to {dict_path}")
        
        # Print summary
        print("\n📋 Data Dictionary Summary:")
        print(f"   Total columns documented: {len(data_dictionary['columns'])}")
        print(f"   Numerical columns: {len([c for c in data_dictionary['columns'].values() if 'mean_value' in c])}")
        print(f"   Categorical columns: {len([c for c in data_dictionary['columns'].values() if 'sample_values' in c])}")
        
        return data_dictionary
    
    def perform_statistical_analysis(self):
        """
        Perform comprehensive statistical analysis of the dataset
        Uses advanced statistics skills
        """
        if self.data is None:
            print("❌ Main dataset not loaded. Please load it first.")
            return
            
        print("\n📊 ADVANCED STATISTICAL ANALYSIS")
        print("=" * 50)
        
        # Correlation Analysis
        print("\n🔗 Correlation Analysis:")
        audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        
        available_features = [f for f in audio_features if f in self.data.columns]
        if available_features:
            correlation_matrix = self.data[available_features + ['popularity']].corr()
            
            # Find strongest correlations with popularity
            pop_corr = correlation_matrix['popularity'].drop('popularity').abs().sort_values(ascending=False)
            print("   Top correlations with popularity:")
            for feature, corr in pop_corr.head(5).items():
                print(f"     {feature}: {corr:.3f}")
        
        # Distribution Analysis
        print("\n📈 Distribution Analysis:")
        for feature in ['popularity', 'energy', 'danceability', 'valence']:
            if feature in self.data.columns:
                # Normality test
                _, p_value = stats.normaltest(self.data[feature].dropna())
                is_normal = "Normal" if p_value > 0.05 else "Not Normal"
                
                print(f"   {feature}:")
                print(f"     Skewness: {stats.skew(self.data[feature].dropna()):.3f}")
                print(f"     Kurtosis: {stats.kurtosis(self.data[feature].dropna()):.3f}")
                print(f"     Distribution: {is_normal} (p={p_value:.4f})")
        
        # Outlier Detection using IQR method
        print("\n🎯 Outlier Detection:")
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outlier_summary[col] = len(outliers)
            
            if len(outliers) > 0:
                print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(self.data)*100:.1f}%)")
        
        return {
            'correlation_matrix': correlation_matrix if available_features else None,
            'outlier_summary': outlier_summary
        }
    
    def create_advanced_visualizations(self):
        """
        Create advanced visualizations using matplotlib and seaborn
        Incorporates data visualization skills
        """
        if self.data is None:
            print("❌ Main dataset not loaded. Please load it first.")
            return
            
        print("\n🎨 CREATING ADVANCED VISUALIZATIONS")
        print("=" * 50)
        
        # Create a comprehensive figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spotify Dataset - Advanced Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Popularity Distribution
        axes[0, 0].hist(self.data['popularity'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Popularity Distribution')
        axes[0, 0].set_xlabel('Popularity Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.data['popularity'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.data["popularity"].mean():.1f}')
        axes[0, 0].legend()
        
        # 2. Audio Features Correlation Heatmap
        audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness']
        available_features = [f for f in audio_features if f in self.data.columns]
        
        if len(available_features) > 3:
            corr_matrix = self.data[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[0, 1], fmt='.2f', square=True)
            axes[0, 1].set_title('Audio Features Correlation')
        
        # 3. Popularity vs Energy Scatter Plot
        if 'energy' in self.data.columns:
            axes[0, 2].scatter(self.data['energy'], self.data['popularity'], 
                             alpha=0.5, s=1, color='green')
            axes[0, 2].set_xlabel('Energy')
            axes[0, 2].set_ylabel('Popularity')
            axes[0, 2].set_title('Popularity vs Energy')
            
            # Add trend line
            z = np.polyfit(self.data['energy'].dropna(), 
                          self.data['popularity'][self.data['energy'].notna()], 1)
            p = np.poly1d(z)
            axes[0, 2].plot(self.data['energy'], p(self.data['energy']), "r--", alpha=0.8)
        
        # 4. Tracks by Decade
        if 'year' in self.data.columns:
            decades = (self.data['year'] // 10) * 10
            decade_counts = decades.value_counts().sort_index()
            
            axes[1, 0].bar(decade_counts.index, decade_counts.values, 
                          color='orange', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Tracks by Decade')
            axes[1, 0].set_xlabel('Decade')
            axes[1, 0].set_ylabel('Number of Tracks')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Audio Features Box Plot
        if len(available_features) > 0:
            # Normalize features for better comparison
            normalized_features = self.data[available_features].copy()
            for feature in available_features:
                normalized_features[feature] = (normalized_features[feature] - 
                                              normalized_features[feature].min()) / \
                                             (normalized_features[feature].max() - 
                                              normalized_features[feature].min())
            
            axes[1, 1].boxplot([normalized_features[feature].dropna() for feature in available_features],
                             labels=available_features)
            axes[1, 1].set_title('Normalized Audio Features Distribution')
            axes[1, 1].set_ylabel('Normalized Value (0-1)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Popularity Trends Over Time
        if 'year' in self.data.columns:
            yearly_popularity = self.data.groupby('year')['popularity'].mean()
            axes[1, 2].plot(yearly_popularity.index, yearly_popularity.values, 
                           color='purple', linewidth=2, marker='o', markersize=3)
            axes[1, 2].set_title('Average Popularity Trends Over Time')
            axes[1, 2].set_xlabel('Year')
            axes[1, 2].set_ylabel('Average Popularity')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spotify_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Advanced visualizations created and saved as 'spotify_analysis_dashboard.png'")
    
    def perform_data_profiling(self):
        """
        Perform comprehensive data profiling
        Incorporates pandas profiling skills
        """
        if self.data is None:
            print("❌ Main dataset not loaded. Please load it first.")
            return
            
        print("\n📋 COMPREHENSIVE DATA PROFILING")
        print("=" * 50)
        
        profile_report = {}
        
        # Basic Dataset Information
        profile_report['basic_info'] = {
            'total_records': len(self.data),
            'total_columns': len(self.data.columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicate_rows': self.data.duplicated().sum(),
            'missing_values_total': self.data.isnull().sum().sum()
        }
        
        # Column-wise Analysis
        profile_report['column_analysis'] = {}
        
        for column in self.data.columns:
            col_info = {
                'dtype': str(self.data[column].dtype),
                'non_null_count': self.data[column].count(),
                'null_count': self.data[column].isnull().sum(),
                'null_percentage': (self.data[column].isnull().sum() / len(self.data)) * 100,
                'unique_values': self.data[column].nunique(),
                'unique_percentage': (self.data[column].nunique() / len(self.data)) * 100
            }
            
            if self.data[column].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': self.data[column].min(),
                    'max': self.data[column].max(),
                    'mean': self.data[column].mean(),
                    'median': self.data[column].median(),
                    'std': self.data[column].std(),
                    'zeros_count': (self.data[column] == 0).sum(),
                    'zeros_percentage': ((self.data[column] == 0).sum() / len(self.data)) * 100
                })
            
            profile_report['column_analysis'][column] = col_info
        
        # Data Quality Assessment
        profile_report['data_quality'] = {
            'completeness_score': (1 - (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns)))) * 100,
            'uniqueness_score': (self.data.duplicated().sum() / len(self.data)) * 100,
            'consistency_score': self._calculate_consistency_score()
        }
        
        # Print Summary
        print("📊 Basic Information:")
        for key, value in profile_report['basic_info'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value:,}")
        
        print("\n🔍 Data Quality Scores:")
        for key, value in profile_report['data_quality'].items():
            print(f"   {key}: {value:.2f}%")
        
        print("\n📋 Column Summary:")
        print(f"   Numerical columns: {len(self.data.select_dtypes(include=[np.number]).columns)}")
        print(f"   Categorical columns: {len(self.data.select_dtypes(include=['object']).columns)}")
        print(f"   Datetime columns: {len(self.data.select_dtypes(include=['datetime64']).columns)}")
        
        # Top Missing Values
        missing_cols = self.data.isnull().sum().sort_values(ascending=False)
        missing_cols = missing_cols[missing_cols > 0]
        
        if len(missing_cols) > 0:
            print("\n❌ Columns with Missing Values:")
            for col, count in missing_cols.head(10).items():
                percentage = (count / len(self.data)) * 100
                print(f"   {col}: {count:,} ({percentage:.1f}%)")
        
        return profile_report
    
    def _calculate_consistency_score(self):
        """
        Calculate a consistency score for the dataset
        """
        consistency_issues = 0
        total_checks = 0
        
        # Check for consistent data types in similar columns
        # Check for reasonable ranges in numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            total_checks += 1
            # Check if values are within reasonable bounds (no extreme outliers)
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Count extreme outliers (beyond 3 * IQR)
            extreme_outliers = self.data[
                (self.data[col] < Q1 - 3 * IQR) | 
                (self.data[col] > Q3 + 3 * IQR)
            ]
            
            if len(extreme_outliers) > len(self.data) * 0.01:  # More than 1% extreme outliers
                consistency_issues += 1
        
        return (1 - (consistency_issues / max(total_checks, 1))) * 100
    
    def generate_insights_report(self):
        """
        Generate automated insights from the data
        Uses advanced analytics skills
        """
        if self.data is None:
            print("❌ Main dataset not loaded. Please load it first.")
            return
            
        print("\n🧠 AUTOMATED INSIGHTS GENERATION")
        print("=" * 50)
        
        insights = []
        
        # Temporal insights
        if 'year' in self.data.columns:
            # Find peak music production years
            yearly_counts = self.data['year'].value_counts().sort_index()
            peak_year = yearly_counts.idxmax()
            peak_count = yearly_counts.max()
            
            insights.append(f"🎵 Peak music production year: {peak_year} with {peak_count:,} tracks")
            
            # Trend analysis
            recent_years = yearly_counts.tail(10)
            if len(recent_years) > 5:
                trend = "increasing" if recent_years.iloc[-1] > recent_years.iloc[0] else "decreasing"
                insights.append(f"📈 Music production trend (last 10 years): {trend}")
        
        # Popularity insights
        if 'popularity' in self.data.columns:
            high_popularity = self.data[self.data['popularity'] > 80]
            if len(high_popularity) > 0:
                insights.append(f"⭐ {len(high_popularity)} tracks ({len(high_popularity)/len(self.data)*100:.1f}%) have high popularity (>80)")
            
            # Find the sweet spot for audio features
            audio_features = ['energy', 'danceability', 'valence']
            for feature in audio_features:
                if feature in self.data.columns:
                    popular_tracks = self.data[self.data['popularity'] > 70]
                    if len(popular_tracks) > 0:
                        avg_feature = popular_tracks[feature].mean()
                        insights.append(f"🎯 Popular tracks tend to have {feature} around {avg_feature:.2f}")
        
        # Artist insights
        if 'artists' in self.data.columns:
            top_artists = self.data['artists'].value_counts().head(5)
            insights.append(f"🎤 Most prolific artist: {top_artists.index[0]} with {top_artists.iloc[0]} tracks")
        
        # Genre insights (if available)
        if self.data_with_genres is not None and 'genres' in self.data_with_genres.columns:
            # Parse genres and find most common
            all_genres = []
            for genre_list in self.data_with_genres['genres'].dropna():
                if isinstance(genre_list, str):
                    # Remove brackets and quotes, split by comma
                    genres = genre_list.strip('[]').replace("'", "").split(', ')
                    all_genres.extend([g.strip() for g in genres if g.strip()])
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts()
                insights.append(f"🎼 Most common genre: {genre_counts.index[0]} ({genre_counts.iloc[0]} occurrences)")
        
        # Audio feature insights
        audio_features = ['valence', 'energy', 'danceability', 'acousticness']
        for feature in audio_features:
            if feature in self.data.columns:
                feature_mean = self.data[feature].mean()
                if feature_mean > 0.7:
                    insights.append(f"🎵 Dataset has high {feature} (avg: {feature_mean:.2f})")
                elif feature_mean < 0.3:
                    insights.append(f"🎵 Dataset has low {feature} (avg: {feature_mean:.2f})")
        
        # Print insights
        print("💡 Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        return insights
    

def main():
    """
    Main function to demonstrate enhanced Phase 1 functionality
    Incorporates advanced data science skills
    """
    print("🎵 SPOTIFY MUSIC RECOMMENDATION SYSTEM - ENHANCED PHASE 1")
    print("=" * 70)
    print("Phase 1: Advanced Data Reading and Comprehensive Analysis")
    print("=" * 70)
    
    # Initialize data reader
    reader = SpotifyDataReader()
    
    # Load and explore all datasets
    print("\n🔄 Loading all datasets...")
    datasets = reader.load_all_datasets()
    
    # Perform comprehensive analysis
    analysis_results = {}
    
    # 1. Basic exploration
    if reader.data is not None:
        print("\n" + "="*50)
        print("1. BASIC DATASET EXPLORATION")
        print("="*50)
        basic_stats = reader.explore_main_dataset()
        analysis_results['basic_stats'] = basic_stats
    
    # 2. Advanced statistical analysis
    if reader.data is not None:
        print("\n" + "="*50)
        print("2. ADVANCED STATISTICAL ANALYSIS")
        print("="*50)
        statistical_analysis = reader.perform_statistical_analysis()
        analysis_results['statistical_analysis'] = statistical_analysis
    
    # 3. Data profiling
    if reader.data is not None:
        print("\n" + "="*50)
        print("3. COMPREHENSIVE DATA PROFILING")
        print("="*50)
        profile_report = reader.perform_data_profiling()
        analysis_results['profile_report'] = profile_report
    
    # 4. Generate insights
    if reader.data is not None:
        print("\n" + "="*50)
        print("4. AUTOMATED INSIGHTS GENERATION")
        print("="*50)
        insights = reader.generate_insights_report()
        analysis_results['insights'] = insights
    
    # 5. Create visualizations
    if reader.data is not None:
        print("\n" + "="*50)
        print("5. ADVANCED VISUALIZATIONS")
        print("="*50)
        reader.create_advanced_visualizations()
    
    # 6. Genre dataset exploration
    if reader.data_with_genres is not None:
        print("\n" + "="*50)
        print("6. GENRE DATASET EXPLORATION")
        print("="*50)
        genre_stats = reader.explore_genre_dataset()
        analysis_results['genre_stats'] = genre_stats
    
    # 7. Create data dictionary
    print("\n" + "="*50)
    print("7. DATA DICTIONARY CREATION")
    print("="*50)
    data_dictionary = reader.create_data_dictionary()
    analysis_results['data_dictionary'] = data_dictionary
    
    # 8. Export data in multiple formats
    print("\n" + "="*50)
    print("8. DATA EXPORT")
    print("="*50)
    reader.export_to_multiple_formats()
    
    # 9. Save comprehensive results
    print("\n" + "="*50)
    print("9. SAVING RESULTS")
    print("="*50)
    reader.save_exploration_results()
    
    # 10. Final summary
    print("\n" + "="*50)
    print("10. FINAL SUMMARY")
    print("="*50)
    summary = reader.create_data_summary_report()
    analysis_results['summary'] = summary
    
    print("\n✅ Enhanced Phase 1 completed successfully!")
    print("📊 Analysis Results Summary:")
    print(f"   📈 Datasets loaded: {len([d for d in datasets.values() if d is not None])}/5")
    print(f"   🎵 Total tracks analyzed: {len(reader.data) if reader.data is not None else 0:,}")
    print(f"   📋 Insights generated: {len(analysis_results.get('insights', []))}")
    print(f"   🎨 Visualizations created: 6 plots")
    print(f"   📚 Data dictionary entries: {len(analysis_results.get('data_dictionary', {}).get('columns', {}))}")
    
    print("\n🔄 Ready for Phase 2: Advanced Exploratory Data Analysis")
    print("💡 Use the analysis results for building the recommendation system!")
    
    return reader, analysis_results


if __name__ == "__main__":
    # Execute Enhanced Phase 1
    print("🚀 Starting Enhanced Phase 1 Analysis...")
    reader, analysis_results = main()
    
    # Additional advanced analysis
    if reader.data is not None:
        print("\n" + "="*50)
        print("🎯 BONUS: QUICK ADVANCED INSIGHTS")
        print("="*50)
        
        # Music evolution insights
        if 'year' in reader.data.columns:
            print("🎵 Music Evolution Analysis:")
            
            # Decade-wise analysis
            decades = (reader.data['year'] // 10) * 10
            decade_stats = reader.data.groupby(decades).agg({
                'popularity': 'mean',
                'energy': 'mean',
                'danceability': 'mean',
                'valence': 'mean'
            }).round(3)
            
            print("   Decade-wise averages:")
            for decade, stats in decade_stats.iterrows():
                print(f"     {decade}s: Pop={stats['popularity']:.2f}, Energy={stats['energy']:.2f}, "
                      f"Dance={stats['danceability']:.2f}, Valence={stats['valence']:.2f}")
        
        # Artist diversity analysis
        if 'artists' in reader.data.columns:
            print(f"\n🎤 Artist Diversity:")
            print(f"   Total unique artists: {reader.data['artists'].nunique():,}")
            print(f"   Average tracks per artist: {len(reader.data) / reader.data['artists'].nunique():.2f}")
            
            # Top prolific artists
            top_artists = reader.data['artists'].value_counts().head(10)
            print("   Top 10 most prolific artists:")
            for artist, count in top_artists.items():
                print(f"     {artist}: {count} tracks")
        
        # Audio feature correlations with popularity
        audio_features = ['valence', 'acousticness', 'danceability', 'energy', 
                         'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
        available_features = [f for f in audio_features if f in reader.data.columns]
        
        if available_features and 'popularity' in reader.data.columns:
            print(f"\n🎼 Audio Features Impact on Popularity:")
            correlations = reader.data[available_features + ['popularity']].corr()['popularity'].drop('popularity')
            correlations = correlations.abs().sort_values(ascending=False)
            
            for feature, corr in correlations.head(5).items():
                direction = "positively" if reader.data[feature].corr(reader.data['popularity']) > 0 else "negatively"
                print(f"   {feature}: {direction} correlated (r={corr:.3f})")
    
    print("\n🎉 Phase 1 Enhanced Analysis Complete!")
    print("📁 Files generated:")
    print("   - data_exploration_results.txt")
    print("   - data_dictionary.json")
    print("   - spotify_analysis_dashboard.png")
    print("   - Various export files (CSV, JSON, Excel, Parquet)")
    print("\n🔄 Ready to proceed to Phase 2: Advanced Exploratory Data Analysis")
