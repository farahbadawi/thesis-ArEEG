# -*- coding: utf-8 -*-
"""
EEG ELECTRODE NECESSITY ANALYSIS
Exploring if all 8 electrodes are necessary for word detection
"""
import warnings
import time
import os
from itertools import combinations
from sklearn.decomposition import FastICA, PCA as SKPCA 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.pipeline import Pipeline 
import pandas as pd 
import numpy as np 
import warnings 
import time 
import joblib 
import os 
import matplotlib.pyplot as plt 
from itertools import combinations 
import scipy.stats as stats
from src.preprocessing import process_eeg, filter_eeg, compute_statistical_features


warnings.filterwarnings("ignore")

print(" " * 20 + "EEG ELECTRODE NECESSITY ANALYSIS")
print("Research Question: Are all 8 electrodes necessary for word detection?")

subject_folder = '/Users/farahbadawi/Desktop/RecordedSessions/sub0'
sampling_freq = 250
test_size = 0.2
random_state = 42

# Electrode analysis settings
min_electrodes = 1
max_electrodes = 8
n_top_combinations = 5  # Number of top combinations to analyze per electrode count

print(f"✓ Subject: {subject_folder}")
print(f"✓ Testing electrode combinations: {min_electrodes} to {max_electrodes} electrodes")
print(f"✓ Will analyze top {n_top_combinations} combinations per electrode count")

# ============================================
# IMPROVED ICA
# ============================================
def improved_ica(X, max_iter=800):
    """Better ICA with component selection"""
    n_trials, n_channels, n_samples = X.shape
    X_clean = np.zeros_like(X)
    
    ica_success = 0
    for trial in range(n_trials):
        try:
            trial_data = X[trial, :, :].T
            trial_data_centered = trial_data - np.mean(trial_data, axis=0)
            
            ica = FastICA(n_components=n_channels, random_state=42, 
                         max_iter=max_iter, tol=1e-4)
            components = ica.fit_transform(trial_data_centered)
            
            if n_channels > 1:
                components[:, 0] = 0
                reconstructed = ica.inverse_transform(components)
            else:
                reconstructed = ica.inverse_transform(components)
            
            X_clean[trial, :, :] = reconstructed.T
            ica_success += 1
            
        except Exception as e:
            X_clean[trial] = X[trial]
    
    print(f"✓ ICA succeeded for {ica_success}/{n_trials} trials")
    return X_clean

# ============================================
# DATA QUALITY VALIDATION
# ============================================
def validate_eeg_data(X, Y):
    """Validate EEG data quality"""
    print("\n" + "="*50)
    print("DATA QUALITY VALIDATION")
    print("="*50)
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {np.unique(Y)}")
    
    label_counts = pd.Series(Y).value_counts()
    print("\nTrial counts per class:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} trials")
    
    # Check channel variances
    print(f"\nChannel variances:")
    for ch in range(X.shape[1]):
        var = np.var(X[:, ch, :])
        print(f"  Electrode {ch}: {var:.6f}")

# ============================================
# FEATURE EXTRACTION FOR ELECTRODE SUBSETS
# ============================================
def extract_features_for_electrodes(X, electrode_indices, sampling_freq=250):
    """Extract features for specific electrode subset"""
    features = []
    n_trials = X.shape[0]
    
    for trial_idx in range(n_trials):
        trial_features = {}
        
        for i, ch in enumerate(electrode_indices):
            sig = X[trial_idx, ch, :]
            
            # Time-domain features
            trial_features[f'mean_ch{i}'] = np.mean(sig)
            trial_features[f'std_ch{i}'] = np.std(sig)
            trial_features[f'var_ch{i}'] = np.var(sig)
            trial_features[f'rms_ch{i}'] = np.sqrt(np.mean(sig**2))
            
            # P300 features
            p300_start = int(0.25 * sampling_freq)
            p300_end = int(0.5 * sampling_freq)
            window = sig[p300_start:p300_end]
            trial_features[f'p300_max_ch{i}'] = np.max(window)
            trial_features[f'p300_mean_ch{i}'] = np.mean(window)
            trial_features[f'p300_auc_ch{i}'] = np.sum(np.abs(window))
            
            # Frequency features
            from scipy import signal
            freqs, psd = signal.welch(sig, fs=sampling_freq, nperseg=min(128, len(sig)))
            delta = (freqs >= 0.5) & (freqs < 4)
            theta = (freqs >= 4) & (freqs < 8)
            alpha = (freqs >= 8) & (freqs < 13)
            beta = (freqs >= 13) & (freqs < 30)
            
            if np.any(delta):
                trial_features[f'delta_power_ch{i}'] = np.mean(psd[delta])
            if np.any(theta):
                trial_features[f'theta_power_ch{i}'] = np.mean(psd[theta])
            if np.any(alpha):
                trial_features[f'alpha_power_ch{i}'] = np.mean(psd[alpha])
            if np.any(beta):
                trial_features[f'beta_power_ch{i}'] = np.mean(psd[beta])
        
        features.append(trial_features)
    
    return pd.DataFrame(features)

# ============================================
# ELECTRODE NECESSITY ANALYSIS
# ============================================
def analyze_electrode_necessity(X_clean, Y):
    """Main analysis: Test all electrode combinations"""
    print("\n" + "="*80)
    print("ELECTRODE NECESSITY ANALYSIS")
    print("="*80)
    print("Testing if all 8 electrodes are necessary for word detection...")
    
    results = []
    all_electrodes = list(range(8))  # Electrodes 0-7
    
    # Test combinations from 1 to 8 electrodes
    for n_elec in range(min_electrodes, max_electrodes + 1):
        print(f"\n🔬 Testing {n_elec} electrode combinations...")
        
        # Get all combinations for this electrode count
        electrode_combinations = list(combinations(all_electrodes, n_elec))
        
        combo_results = []
        for i, electrode_combo in enumerate(electrode_combinations):
            if i % 10 == 0:  # Progress indicator
                print(f"  Progress: {i}/{len(electrode_combinations)}")
            
            try:
                # Extract features for this electrode combination
                features_df = extract_features_for_electrodes(X_clean, electrode_combo)
                X_feat = features_df.values
                
                # Scale features
                scaler = StandardScaler()
                X_feat_scaled = scaler.fit_transform(X_feat)
                
                # Train/test split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_feat_scaled, Y, test_size=test_size, 
                    random_state=random_state, stratify=Y
                )
                
                # Test multiple classifiers
                classifiers = {
                    'RF': RandomForestClassifier(n_estimators=100, random_state=random_state),
                    'SVM': SVC(kernel='rbf', random_state=random_state),
                    'MLP': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=random_state)
                }
                
                accuracies = {}
                for name, clf in classifiers.items():
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)
                    accuracies[name] = accuracy_score(y_te, y_pred)
                
                # Store results
                combo_results.append({
                    'electrode_combo': electrode_combo,
                    'num_electrodes': n_elec,
                    'RF_accuracy': accuracies['RF'],
                    'SVM_accuracy': accuracies['SVM'],
                    'MLP_accuracy': accuracies['MLP'],
                    'mean_accuracy': np.mean(list(accuracies.values()))
                })
                
            except Exception as e:
                print(f"Error with electrodes {electrode_combo}: {e}")
                continue
        
        # Get top combinations for this electrode count
        combo_results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
        results.extend(combo_results[:n_top_combinations])
        
        print(f"✓ Top {n_top_combinations} combinations for {n_elec} electrodes:")
        for i, res in enumerate(combo_results[:n_top_combinations]):
            print(f"    {i+1}. Electrodes {res['electrode_combo']}: {res['mean_accuracy']:.3f}")
    
    return pd.DataFrame(results)

# ============================================
# VISUALIZATION AND ANALYSIS
# ============================================
def visualize_electrode_analysis(results_df):
    """Create comprehensive visualizations"""
    print("\n" + "="*80)
    print("VISUALIZATION & ANALYSIS")
    print("="*80)
    
    # 1. Performance vs Number of Electrodes
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean accuracy vs electrode count
    plt.subplot(2, 3, 1)
    electrode_stats = results_df.groupby('num_electrodes').agg({
        'mean_accuracy': ['mean', 'max', 'min'],
        'RF_accuracy': 'mean',
        'SVM_accuracy': 'mean', 
        'MLP_accuracy': 'mean'
    }).round(3)
    
    plt.plot(electrode_stats.index, electrode_stats[('mean_accuracy', 'mean')], 
             marker='o', linewidth=2, label='Mean Accuracy')
    plt.plot(electrode_stats.index, electrode_stats[('mean_accuracy', 'max')], 
             marker='s', linestyle='--', label='Max Accuracy')
    plt.xlabel('Number of Electrodes')
    plt.ylabel('Accuracy')
    plt.title('Performance vs Number of Electrodes')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Classifier comparison
    plt.subplot(2, 3, 2)
    classifier_means = results_df.groupby('num_electrodes').agg({
        'RF_accuracy': 'mean',
        'SVM_accuracy': 'mean',
        'MLP_accuracy': 'mean'
    })
    
    plt.plot(classifier_means.index, classifier_means['RF_accuracy'], 
             marker='o', label='Random Forest')
    plt.plot(classifier_means.index, classifier_means['SVM_accuracy'], 
             marker='s', label='SVM')
    plt.plot(classifier_means.index, classifier_means['MLP_accuracy'], 
             marker='^', label='MLP')
    plt.xlabel('Number of Electrodes')
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Electrode frequency in top combinations
    plt.subplot(2, 3, 3)
    electrode_usage = {}
    for combo in results_df['electrode_combo']:
        for electrode in combo:
            electrode_usage[electrode] = electrode_usage.get(electrode, 0) + 1
    
    electrodes = list(electrode_usage.keys())
    usage_counts = [electrode_usage[e] for e in electrodes]
    
    plt.bar(electrodes, usage_counts)
    plt.xlabel('Electrode Number')
    plt.ylabel('Frequency in Top Combinations')
    plt.title('Electrode Importance')
    
    # Plot 4: Performance distribution by electrode count
    plt.subplot(2, 3, 4)
    accuracy_data = []
    labels = []
    for n_elec in range(min_electrodes, max_electrodes + 1):
        subset = results_df[results_df['num_electrodes'] == n_elec]
        accuracy_data.append(subset['mean_accuracy'].values)
        labels.append(f'{n_elec} electrodes')
    
    plt.boxplot(accuracy_data, labels=labels)
    plt.xlabel('Number of Electrodes')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Distribution by Electrode Count')
    plt.xticks(rotation=45)
    
    # Plot 5: Best combination for each electrode count
    plt.subplot(2, 3, 5)
    best_by_count = results_df.loc[results_df.groupby('num_electrodes')['mean_accuracy'].idxmax()]
    plt.bar(best_by_count['num_electrodes'], best_by_count['mean_accuracy'])
    plt.xlabel('Number of Electrodes')
    plt.ylabel('Best Accuracy')
    plt.title('Best Performance by Electrode Count')
    
    # Plot 6: Performance gain per additional electrode
    plt.subplot(2, 3, 6)
    performance_gain = []
    for i in range(1, len(best_by_count)):
        gain = best_by_count.iloc[i]['mean_accuracy'] - best_by_count.iloc[i-1]['mean_accuracy']
        performance_gain.append(gain)
    
    plt.bar(range(2, max_electrodes + 1), performance_gain)
    plt.xlabel('Number of Electrodes')
    plt.ylabel('Accuracy Gain')
    plt.title('Performance Gain per Additional Electrode')
    
    plt.tight_layout()
    plt.savefig('electrode_necessity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return electrode_stats, best_by_count

# ============================================
# FINAL RECOMMENDATIONS
# ============================================
def generate_recommendations(results_df, electrode_stats, best_by_count):
    """Generate practical recommendations"""
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    # Find optimal electrode count (where adding more electrodes gives diminishing returns)
    performance_gains = []
    for i in range(1, len(best_by_count)):
        current_acc = best_by_count.iloc[i]['mean_accuracy']
        prev_acc = best_by_count.iloc[i-1]['mean_accuracy']
        gain = current_acc - prev_acc
        performance_gains.append((best_by_count.iloc[i]['num_electrodes'], gain))
    
    # Find point of diminishing returns (gain < 0.02)
    optimal_count = max_electrodes
    for count, gain in performance_gains:
        if gain < 0.02:  # Less than 2% improvement
            optimal_count = count
            break
    
    best_overall = results_df.loc[results_df['mean_accuracy'].idxmax()]
    
    print(f"🏆 BEST OVERALL PERFORMANCE:")
    print(f"   Electrodes: {best_overall['electrode_combo']}")
    print(f"   Count: {best_overall['num_electrodes']}")
    print(f"   Accuracy: {best_overall['mean_accuracy']:.3f}")
    
    print(f"\n🎯 OPTIMAL ELECTRODE COUNT: {optimal_count} electrodes")
    print(f"   Reason: Adding more electrodes provides diminishing returns")
    
    best_optimal = best_by_count[best_by_count['num_electrodes'] == optimal_count].iloc[0]
    print(f"   Best {optimal_count}-electrode combination: {best_optimal['electrode_combo']}")
    print(f"   Accuracy: {best_optimal['mean_accuracy']:.3f}")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    for n_elec in range(min_electrodes, max_electrodes + 1):
        stats = electrode_stats.loc[n_elec]
        best_combo = best_by_count[best_by_count['num_electrodes'] == n_elec].iloc[0]
        print(f"   {n_elec} electrodes: Mean={stats[('mean_accuracy', 'mean')]:.3f}, "
              f"Best={best_combo['mean_accuracy']:.3f} ({best_combo['electrode_combo']})")
    
    print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
    if optimal_count <= 3:
        print("   ✅ EXCELLENT: You can achieve good performance with minimal electrodes!")
        print("   → Consider using only the optimal electrode set for practical applications")
    elif optimal_count <= 5:
        print("   ✅ GOOD: Balanced performance with reasonable number of electrodes")
        print("   → Good trade-off between accuracy and practicality")
    else:
        print("   ⚠️  MODERATE: Requires more electrodes for best performance")
        print("   → Consider if the accuracy gain justifies the additional setup complexity")
    
    print(f"\n🔧 SUGGESTED NEXT STEPS:")
    print(f"   1. Validate optimal electrode set ({best_optimal['electrode_combo']}) on other subjects")
    print(f"   2. Test cross-subject generalization")
    print(f"   3. Consider practical constraints (comfort, setup time)")
    print(f"   4. Explore if different electrode placements work better")

# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """Main execution function"""
    print("🚀 STARTING EEG ELECTRODE NECESSITY ANALYSIS...")
    
    # 1. Load and preprocess data
    print("\n[1/4] Loading and preprocessing EEG data...")
    X, Y = process_eeg(
        TIME_STEPS=1200,
        included_states=["Up", "Down", "Left", "Right", "Select"],
        subject_folder=subject_folder
    )
    
    validate_eeg_data(X, Y)
    
    print("\n[2/4] Filtering and artifact removal...")
    X_filtered = filter_eeg(X, sampling_freq=250)
    X_clean = improved_ica(X_filtered)
    print(f"✓ Preprocessing complete: {X_clean.shape}")
    
    # 2. Electrode necessity analysis
    print("\n[3/4] Performing electrode necessity analysis...")
    results_df = analyze_electrode_necessity(X_clean, Y)
    
    # Save results
    results_df.to_csv('electrode_necessity_results.csv', index=False)
    print("✓ Results saved to 'electrode_necessity_results.csv'")
    
    # 3. Visualization and analysis
    print("\n[4/4] Generating visualizations and recommendations...")
    electrode_stats, best_by_count = visualize_electrode_analysis(results_df)
    
    # 4. Generate recommendations
    generate_recommendations(results_df, electrode_stats, best_by_count)
    
    print("\n" + "="*80)
    print("✅ ELECTRODE NECESSITY ANALYSIS COMPLETE!")
    print("="*80)
    print("\n📁 Output files:")
    print("   • electrode_necessity_results.csv - All results")
    print("   • electrode_necessity_analysis.png - Comprehensive visualizations")
    print("\n🎯 Research question answered: Are all 8 electrodes necessary?")
    print("   → See recommendations above for the optimal electrode count!")

# ============================================
# RUN THE ANALYSIS
# ============================================
if __name__ == "__main__":
    main()