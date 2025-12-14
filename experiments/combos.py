# -*- coding: utf-8 -*-
"""
COMPLETE EEG FEATURE EXTRACTION & ML PIPELINE - ENHANCED VERSION
With Data Quality Checks, Improved ICA, and Cross-Subject Validation
"""
from sklearn.decomposition import FastICA, PCA as SKPCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import pandas as pd
import numpy as np
import warnings
import time
import joblib
import os
import scipy.stats as stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print("ENHANCED EEG PIPELINE - WITH VALIDATION")

print("CONFIGURATION")

# Data paths
subject_folder = '/Users/farahbadawi/Desktop/RecordedSessions/sub0'
test_subject_folder = '/Users/farahbadawi/Desktop/RecordedSessions/sub0' #for cross-validation

# Hyperparameter search settings
use_randomized = True
randomized_iters = 40
cv_folds = 5

# PCA settings
pca_mode = "var"
variance_ratio = 0.95
n_pca_components = 50

# Output
save_models_dir = "best_models_enhanced"
os.makedirs(save_models_dir, exist_ok=True)

print(f"✓ Training subject: {subject_folder}")
print(f"✓ Test subject: {test_subject_folder}")
print(f"✓ Hyperparameter search: {'Randomized' if use_randomized else 'Grid'}")
print(f"✓ CV folds: {cv_folds}")

# ENHANCED DATA VALIDATION
def enhanced_data_validation(X, Y):
    print("\n" + "="*50)
    print("DATA QUALITY ANALYSIS")
    print("="*50)
    
    # Check trial counts per class
    label_counts = pd.Series(Y).value_counts()
    print("Trial counts per class:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} trials")
    
    # Check data statistics
    print(f"\nData statistics:")
    print(f"  Shape: {X.shape}")
    print(f"  Range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Mean: {X.mean():.3f} ± {X.std():.3f}")
    
    # Check for NaN/Infinite values
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  NaN values: {nan_count}, Infinite values: {inf_count}")
    
    # Check channel variances
    print(f"\nChannel variances:")
    low_variance_channels = []
    for ch in range(X.shape[1]):
        var = np.var(X[:, ch, :])
        print(f"  Channel {ch}: {var:.6f}")
        if var < 1e-6:
            print(f"    ⚠️ LOW VARIANCE - possible dead channel!")
            low_variance_channels.append(ch)
    
    return label_counts, low_variance_channels

# IMPROVED ICA
def improved_ica(X, max_iter=800):
    """Better ICA with component selection"""
    n_trials, n_channels, n_samples = X.shape
    X_clean = np.zeros_like(X)
    
    ica_success = 0
    for trial in range(n_trials):
        try:
            # Transpose to samples x channels
            trial_data = X[trial, :, :].T
            
            # Center the data
            trial_data_centered = trial_data - np.mean(trial_data, axis=0)
            
            ica = FastICA(n_components=n_channels, random_state=42, 
                         max_iter=max_iter, tol=1e-4)
            components = ica.fit_transform(trial_data_centered)
            
            # Reconstruct without the first component (often artifacts)
            if n_channels > 1:
                components[:, 0] = 0  # Remove first component
                reconstructed = ica.inverse_transform(components)
            else:
                reconstructed = ica.inverse_transform(components)
            
            X_clean[trial, :, :] = reconstructed.T
            ica_success += 1
            
        except Exception as e:
            # print(f"ICA failed for trial {trial}: {e}")
            X_clean[trial] = X[trial]  # Use original if ICA fails
    
    print(f"✓ ICA succeeded for {ica_success}/{n_trials} trials")
    return X_clean

# CORRELATION-BASED FEATURE SELECTION
def remove_correlated_features(feature_df, threshold=0.95):
    """Remove highly correlated features"""
    corr_matrix = feature_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper_tri.columns 
               if any(upper_tri[column] > threshold)]
    
    print(f"Removing {len(to_drop)} highly correlated features (r > {threshold})")
    return feature_df.drop(columns=to_drop)

# CROSS-SUBJECT VALIDATION
def cross_subject_validation(best_model, test_subject_folder, feature_combo_keys):
    """Test best model on a different subject"""
    print(f"\n🧪 CROSS-SUBJECT VALIDATION: {test_subject_folder}")
    
    # Load test subject data
    X_test, Y_test = process_eeg(
        TIME_STEPS=1200,
        included_states=["Up", "Down", "Left", "Right", "Select"],
        subject_folder=test_subject_folder
    )
    
    # Preprocess same as training
    X_test_filtered = filter_eeg(X_test)
    X_test_clean = improved_ica(X_test_filtered)
    
    # Extract features using the same combination as best model
    feature_functions = {
        'P300': extract_p300_features,
        'Statistical': lambda X: compute_statistical_features(X, axis='time'),
        'TimeFreq': extract_timefreq_features,
        'Spatial': extract_spatial_features,
        'CSP': lambda X: extract_csp_features(X, Y_test),
        'Riemannian': extract_riemannian_features
    }
    
    # Build the same feature combination
    feature_dfs = []
    for key in feature_combo_keys:
        if key in feature_functions:
            feature_dfs.append(feature_functions[key](X_test_clean))
    
    X_test_combo = pd.concat(feature_dfs, axis=1)
    X_test_combo = X_test_combo.select_dtypes(include=[np.number])
    
    # Predict
    y_pred = best_model.predict(X_test_combo.values)
    accuracy = accuracy_score(Y_test, y_pred)
    
    print(f"Cross-subject accuracy: {accuracy:.3f}")
    print(classification_report(Y_test, y_pred))
    
    return accuracy

# FEATURE IMPORTANCE ANALYSIS
def analyze_best_features(best_model, feature_names, top_n=20):
    """Analyze which features are most important in the best model"""
    if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        importances = best_model.named_steps['clf'].feature_importances_
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 TOP {top_n} MOST IMPORTANT FEATURES:")
        print(feat_imp.head(top_n).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = feat_imp.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feat_imp
    else:
        print("Feature importance not available for this model type")
        return None

# ============================================
# ENHANCED RESULTS ANALYSIS
# ============================================
def enhanced_results_analysis(all_results):
    print("\n" + "="*80)
    print("ENHANCED RESULTS ANALYSIS")
    print("="*80)
    
    # Best model overall
    best_overall = all_results.iloc[0]
    print(f"🏆 BEST OVERALL MODEL:")
    print(f"   Combo: {best_overall['combo']}")
    print(f"   Model: {best_overall['model']}")
    print(f"   PCA: {best_overall['pca']}")
    print(f"   Test Accuracy: {best_overall['test_acc']:.3f}")
    
    # Analysis by model type
    print(f"\n📊 PERFORMANCE BY MODEL TYPE:")
    model_stats = all_results.groupby('model').agg({
        'test_acc': ['mean', 'max', 'min', 'std']
    }).round(3)
    print(model_stats)
    
    # Analysis by feature type
    print(f"\n🎯 PERFORMANCE BY FEATURE COMBO (Top 5):")
    combo_stats = all_results.groupby('combo').agg({
        'test_acc': 'max'
    }).sort_values('test_acc', ascending=False).head(5)
    print(combo_stats)
    
    # PCA vs No PCA
    pca_stats = all_results.groupby('pca').agg({
        'test_acc': 'mean'
    }).round(3)
    print(f"\n🔍 PCA IMPACT:")
    print(f"   With PCA: {pca_stats.loc[True, 'test_acc']}")
    print(f"   Without PCA: {pca_stats.loc[False, 'test_acc']}")

# ============================================
# PART 1: PREPROCESSING WITH VALIDATION
# ============================================
print("\n" + "="*80)
print("PART 1: PREPROCESSING WITH DATA VALIDATION")
print("="*80)

print("\n[1/3] Loading raw data...")
X, Y = process_eeg(
    TIME_STEPS=1200,
    included_states=["Up", "Down", "Left", "Right", "Select"],
    subject_folder=subject_folder
)
print(f"✓ Data loaded: {X.shape}")

# Enhanced data validation
label_counts, low_var_channels = enhanced_data_validation(X, Y)
if low_var_channels:
    print(f"⚠️  WARNING: Low variance in channels {low_var_channels}")

print("\n[2/3] Filtering data...")
X_filtered = filter_eeg(X, sampling_freq=250)
print(f"✓ Filtered: {X_filtered.shape}")

print("\n[3/3] Applying IMPROVED ICA for artifact removal...")
X_clean = improved_ica(X_filtered)
print(f"✓ ICA complete: {X_clean.shape}")

# Save preprocessed data
np.save('X_clean_enhanced.npy', X_clean)
np.save('Y_labels_enhanced.npy', Y)
print(f"✓ Saved preprocessed data")

# ============================================
# PART 2: FEATURE EXTRACTION (All Methods)
# ============================================
print("\n" + "="*80)
print("PART 2: FEATURE EXTRACTION (6 METHODS)")
print("="*80)

# ----------------------------------------
# 2.1: P300/ERP Features
# ----------------------------------------
print("\n[1/6] Extracting P300/ERP features...")

def extract_p300_features(X, sampling_freq=250):
    p300_start = int(0.25 * sampling_freq)
    p300_end = int(0.5 * sampling_freq)
    
    features = []
    for trial_idx in range(X.shape[0]):
        trial_features = {}
        for ch in range(X.shape[1]):
            window = X[trial_idx, ch, p300_start:p300_end]
            trial_features[f'p300_amp_ch{ch+1}'] = np.max(window)
            peak_sample = np.argmax(window) + p300_start
            trial_features[f'p300_lat_ch{ch+1}'] = (peak_sample / sampling_freq) * 1000
            trial_features[f'p300_mean_ch{ch+1}'] = np.mean(window)
            trial_features[f'p300_auc_ch{ch+1}'] = np.sum(np.abs(window))
        features.append(trial_features)
    return pd.DataFrame(features)

p300_features = extract_p300_features(X_clean)
print(f"✓ P300 features: {p300_features.shape}")

# ----------------------------------------
# 2.2: Statistical Features
# ----------------------------------------
print("\n[2/6] Extracting statistical features...")
stat_features = compute_statistical_features(X_clean, axis='time')
print(f"✓ Statistical features: {stat_features.shape}")

# ----------------------------------------
# 2.3: Time-Frequency Features
# ----------------------------------------
print("\n[3/6] Extracting time-frequency features...")

def extract_timefreq_features(X, sampling_freq=250):
    from scipy import signal as sig
    
    features = []
    for trial_idx in range(X.shape[0]):
        trial_features = {}
        for ch in range(X.shape[1]):
            eeg_signal = X[trial_idx, ch, :]
            
            # Compute power spectral density
            freqs, psd = sig.welch(eeg_signal, fs=sampling_freq, nperseg=128)
            
            # Define frequency bands
            delta = (freqs >= 0.5) & (freqs < 4)
            theta = (freqs >= 4) & (freqs < 8)
            alpha = (freqs >= 8) & (freqs < 13)
            beta = (freqs >= 13) & (freqs < 30)
            
            # Power in each band
            trial_features[f'delta_power_ch{ch+1}'] = np.mean(psd[delta])
            trial_features[f'theta_power_ch{ch+1}'] = np.mean(psd[theta])
            trial_features[f'alpha_power_ch{ch+1}'] = np.mean(psd[alpha])
            trial_features[f'beta_power_ch{ch+1}'] = np.mean(psd[beta])
            
            # Relative power
            total_power = np.sum(psd)
            trial_features[f'delta_rel_ch{ch+1}'] = np.sum(psd[delta]) / total_power
            trial_features[f'alpha_rel_ch{ch+1}'] = np.sum(psd[alpha]) / total_power
            
        features.append(trial_features)
    return pd.DataFrame(features)

timefreq_features = extract_timefreq_features(X_clean)
print(f"✓ Time-frequency features: {timefreq_features.shape}")

# ----------------------------------------
# 2.4: Spatial Features
# ----------------------------------------
print("\n[4/6] Extracting spatial features...")

def extract_spatial_features(X, sampling_freq=250):
    p300_start = int(0.25 * sampling_freq)
    p300_end = int(0.5 * sampling_freq)
    
    features = []
    for trial_idx in range(X.shape[0]):
        trial_features = {}
        p300_all = X[trial_idx, :, p300_start:p300_end]
        
        trial_features['p300_max_global'] = np.max(p300_all)
        trial_features['p300_mean_global'] = np.mean(p300_all)
        
        channel_peaks = np.max(p300_all, axis=1)
        trial_features['p300_peak_channel'] = np.argmax(channel_peaks)
        trial_features['p300_channel_var'] = np.var(channel_peaks)
        
        if X.shape[1] > 1:
            corr = np.corrcoef(p300_all)
            upper_tri = corr[np.triu_indices_from(corr, k=1)]
            trial_features['p300_avg_corr'] = np.mean(upper_tri)
        
        features.append(trial_features)
    return pd.DataFrame(features)

spatial_features = extract_spatial_features(X_clean)
print(f"✓ Spatial features: {spatial_features.shape}")

# ----------------------------------------
# 2.5: Common Spatial Pattern (CSP) Features
# ----------------------------------------
print("\n[5/6] Extracting CSP features...")

def extract_csp_features(X, y, n_components=4):
    csp_features_list = []
    classes = np.unique(y)
    
    for trial_idx in range(X.shape[0]):
        trial_features = {}
        
        for i, class1 in enumerate(classes):
            for class2 in classes[i+1:]:
                mask = (y == class1) | (y == class2)
                X_binary = X[mask]
                y_binary = y[mask]
                
                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
                try:
                    csp.fit(X_binary, y_binary)
                    trial_data = X[trial_idx:trial_idx+1]
                    csp_transformed = csp.transform(trial_data)
                    
                    for comp in range(n_components):
                        trial_features[f'csp_{class1}vs{class2}_comp{comp+1}'] = csp_transformed[0, comp]
                except:
                    for comp in range(n_components):
                        trial_features[f'csp_{class1}vs{class2}_comp{comp+1}'] = 0.0
        
        csp_features_list.append(trial_features)
    
    return pd.DataFrame(csp_features_list)

csp_features = extract_csp_features(X_clean, Y, n_components=4)
print(f"✓ CSP features: {csp_features.shape}")

# ----------------------------------------
# 2.6: Riemannian Geometry Features
# ----------------------------------------
print("\n[6/6] Extracting Riemannian geometry features...")

def extract_riemannian_features(X):
    cov_estimator = Covariances(estimator='lwf')
    cov_matrices = cov_estimator.fit_transform(X)
    
    ts = TangentSpace(metric='riemann')
    riemannian_features = ts.fit_transform(cov_matrices)
    
    n_features = riemannian_features.shape[1]
    columns = [f'riemann_{i+1}' for i in range(n_features)]
    return pd.DataFrame(riemannian_features, columns=columns)

riemannian_features = extract_riemannian_features(X_clean)
print(f"✓ Riemannian features: {riemannian_features.shape}")

# ============================================
# PART 3: ORGANIZE FEATURES WITH CORRELATION REMOVAL
# ============================================
print("\n" + "="*80)
print("PART 3: ORGANIZING FEATURES WITH CORRELATION REMOVAL")
print("="*80)

# Store all feature groups
feature_groups = {
    'P300': p300_features,
    'Statistical': stat_features,
    'TimeFreq': timefreq_features,
    'Spatial': spatial_features,
    'CSP': csp_features,
    'Riemannian': riemannian_features
}

# Define PRIORITY feature combinations (based on EEG literature)
priority_combos = {
    'CSP_Riemannian': ['CSP', 'Riemannian'],  # Usually best for motor imagery
    'TimeFreq_Stat': ['TimeFreq', 'Statistical'],  # Good for ERP
    'ERP_CSP': ['P300', 'CSP'],  # Good for P300 paradigms
    'Riemannian_TimeFreq': ['Riemannian', 'TimeFreq'],  # Modern approach
    'All_6': ['P300', 'Statistical', 'TimeFreq', 'Spatial', 'CSP', 'Riemannian'],
    'CSP_Only': ['CSP'],
    'Riemannian_Only': ['Riemannian'],
    'TimeFreq_Only': ['TimeFreq']
}

print(f"\nFeature groups available: {list(feature_groups.keys())}")
print(f"Priority combinations to test: {len(priority_combos)}")

# Helper function to build X from combo with correlation removal
def build_X_from_combo(combo_keys, correlation_threshold=0.95):
    dfs = [feature_groups[k].reset_index(drop=True) for k in combo_keys]
    X_df = pd.concat(dfs, axis=1)
    X_df = X_df.select_dtypes(include=[np.number])
    
    # Remove highly correlated features
    X_df_filtered = remove_correlated_features(X_df, threshold=correlation_threshold)
    
    return X_df_filtered.values, X_df_filtered.columns.tolist()

# Save all features combined
all_features = pd.concat([
    p300_features,
    stat_features,
    timefreq_features,
    spatial_features,
    csp_features,
    riemannian_features
], axis=1)
all_features['Label'] = Y
all_features.to_csv('all_features_enhanced.csv', index=False)
print(f"\n✓ Saved all features to 'all_features_enhanced.csv'")
print(f"  Total features: {all_features.shape[1] - 1} (+ label)")

# ============================================
# PART 4: HYPERPARAMETER SEARCH - RF & SVM
# ============================================
print("\n" + "="*80)
print("PART 4: HYPERPARAMETER SEARCH - RANDOM FOREST & SVM")
print("="*80)

# Define hyperparameter distributions
rf_dist = {
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__bootstrap': [True, False]
}

svm_dist = {
    'clf__C': stats.loguniform(1e-2, 1e2),
    'clf__gamma': ['scale', 'auto', 1e-3, 1e-2],
    'clf__kernel': ['rbf', 'linear']
}

results_rf_svm = []
y_all = Y
feature_names_dict = {}  # Store feature names for each combo

for combo_name, combo_keys in priority_combos.items():
    try:
        print("\n" + "-"*70)
        print(f"PRIORITY COMBO: {combo_name} -> {combo_keys}")
        print("-"*70)
        
        X_combo, feature_names = build_X_from_combo(combo_keys)
        feature_names_dict[combo_name] = feature_names
        print(f"  Shape after correlation removal: {X_combo.shape}")
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_combo, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        # Test both with and without PCA
        for pca_flag in ['no_pca', 'pca']:
            if pca_flag == 'pca':
                if pca_mode == 'var':
                    pca_step = ('pca', SKPCA(n_components=variance_ratio, svd_solver='full'))
                else:
                    pca_step = ('pca', SKPCA(n_components=n_pca_components))
                print(f"  → Testing WITH PCA")
            else:
                pca_step = None
                print(f"  → Testing WITHOUT PCA")
            
            # Random Forest
            steps_rf = [('scaler', StandardScaler())]
            if pca_step:
                steps_rf.append(pca_step)
            steps_rf.append(('clf', RandomForestClassifier(random_state=42)))
            pipe_rf = Pipeline(steps_rf)
            
            if use_randomized:
                rf_search = RandomizedSearchCV(
                    pipe_rf, rf_dist, n_iter=randomized_iters, cv=cv_folds,
                    n_jobs=-1, verbose=0, scoring='accuracy', random_state=42
                )
            else:
                rf_search = GridSearchCV(
                    pipe_rf, rf_dist, cv=cv_folds,
                    n_jobs=-1, verbose=0, scoring='accuracy'
                )
            
            t0 = time.time()
            rf_search.fit(X_tr, y_tr)
            rf_time = time.time() - t0
            
            best_rf = rf_search.best_estimator_
            rf_cv_best = rf_search.best_score_
            y_pred_rf = best_rf.predict(X_te)
            rf_test_acc = accuracy_score(y_te, y_pred_rf)
            
            print(f"    RF: CV={rf_cv_best:.3f} | Test={rf_test_acc:.3f} | {rf_time:.1f}s")
            
            # Save model
            rf_path = os.path.join(save_models_dir, f"{combo_name}__{pca_flag}__RF.pkl")
            joblib.dump(best_rf, rf_path)
            
            # SVM
            steps_svm = [('scaler', StandardScaler())]
            if pca_step:
                steps_svm.append(pca_step)
            steps_svm.append(('clf', SVC(random_state=42)))
            pipe_svm = Pipeline(steps_svm)
            
            if use_randomized:
                svm_search = RandomizedSearchCV(
                    pipe_svm, svm_dist, n_iter=randomized_iters, cv=cv_folds,
                    n_jobs=-1, verbose=0, scoring='accuracy', random_state=42
                )
            else:
                svm_search = GridSearchCV(
                    pipe_svm, svm_dist, cv=cv_folds,
                    n_jobs=-1, verbose=0, scoring='accuracy'
                )
            
            t0 = time.time()
            svm_search.fit(X_tr, y_tr)
            svm_time = time.time() - t0
            
            best_svm = svm_search.best_estimator_
            svm_cv_best = svm_search.best_score_
            y_pred_svm = best_svm.predict(X_te)
            svm_test_acc = accuracy_score(y_te, y_pred_svm)
            
            print(f"    SVM: CV={svm_cv_best:.3f} | Test={svm_test_acc:.3f} | {svm_time:.1f}s")
            
            # Save model
            svm_path = os.path.join(save_models_dir, f"{combo_name}__{pca_flag}__SVM.pkl")
            joblib.dump(best_svm, svm_path)
            
            # Store results
            results_rf_svm.append({
                'combo': combo_name,
                'features': "+".join(combo_keys),
                'pca': (pca_flag == 'pca'),
                'model': 'RF',
                'cv_best': rf_cv_best,
                'test_acc': rf_test_acc,
                'best_params': rf_search.best_params_,
                'time_s': rf_time
            })
            
            results_rf_svm.append({
                'combo': combo_name,
                'features': "+".join(combo_keys),
                'pca': (pca_flag == 'pca'),
                'model': 'SVM',
                'cv_best': svm_cv_best,
                'test_acc': svm_test_acc,
                'best_params': svm_search.best_params_,
                'time_s': svm_time
            })
            
    except Exception as e:
        print(f"  ERROR: {e}")
        results_rf_svm.append({'combo': combo_name, 'error': str(e)})

# Save RF & SVM results
df_rf_svm = pd.DataFrame(results_rf_svm)
df_rf_svm = df_rf_svm.sort_values(by='test_acc', ascending=False)
df_rf_svm.to_csv('results_RF_SVM_enhanced.csv', index=False)
print(f"\n✓ RF & SVM results saved to 'results_RF_SVM_enhanced.csv'")

# ============================================
# PART 5: HYPERPARAMETER SEARCH - NEURAL NETWORK
# ============================================
print("\n" + "="*80)
print("PART 5: HYPERPARAMETER SEARCH - NEURAL NETWORK (MLP)")
print("="*80)

mlp_params = {
    'clf__hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
    'clf__activation': ['relu', 'tanh'],
    'clf__alpha': [0.0001, 0.001, 0.01],
    'clf__learning_rate_init': [0.001, 0.01],
    'clf__max_iter': [500]
}

results_mlp = []

for combo_name, combo_keys in priority_combos.items():
    try:
        print("\n" + "-"*70)
        print(f"PRIORITY COMBO: {combo_name} -> {combo_keys}")
        print("-"*70)
        
        X_combo, feature_names = build_X_from_combo(combo_keys)
        print(f"  Shape after correlation removal: {X_combo.shape}")
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_combo, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        
        for pca_flag in ['no_pca', 'pca']:
            if pca_flag == 'pca':
                if pca_mode == 'var':
                    pca_step = ('pca', SKPCA(n_components=variance_ratio, svd_solver='full'))
                else:
                    pca_step = ('pca', SKPCA(n_components=n_pca_components))
                print(f"  → Testing WITH PCA")
            else:
                pca_step = None
                print(f"  → Testing WITHOUT PCA")
            
            steps_mlp = [('scaler', StandardScaler())]
            if pca_step:
                steps_mlp.append(pca_step)
            steps_mlp.append(('clf', MLPClassifier(random_state=42)))
            pipe_mlp = Pipeline(steps_mlp)
            
            if use_randomized:
                mlp_search = RandomizedSearchCV(
                    pipe_mlp, mlp_params, n_iter=randomized_iters, cv=cv_folds,
                    n_jobs=-1, verbose=0, scoring='accuracy', random_state=42
                )
            else:
                mlp_search = GridSearchCV(
                    pipe_mlp, mlp_params, cv=cv_folds,
                    n_jobs=-1, verbose=0, scoring='accuracy'
                )
            
            t0 = time.time()
            mlp_search.fit(X_tr, y_tr)
            mlp_time = time.time() - t0
            
            best_mlp = mlp_search.best_estimator_
            mlp_cv_best = mlp_search.best_score_
            y_pred_mlp = best_mlp.predict(X_te)
            mlp_test_acc = accuracy_score(y_te, y_pred_mlp)
            
            print(f"    MLP: CV={mlp_cv_best:.3f} | Test={mlp_test_acc:.3f} | {mlp_time:.1f}s")
            
            # Save model
            mlp_path = os.path.join(save_models_dir, f"{combo_name}__{pca_flag}__MLP.pkl")
            joblib.dump(best_mlp, mlp_path)
            
            results_mlp.append({
                'combo': combo_name,
                'features': "+".join(combo_keys),
                'pca': (pca_flag == 'pca'),
                'model': 'MLP',
                'cv_best': mlp_cv_best,
                'test_acc': mlp_test_acc,
                'best_params': mlp_search.best_params_,
                'time_s': mlp_time
            })
            
    except Exception as e:
        print(f"  ERROR: {e}")
        results_mlp.append({'combo': combo_name, 'error': str(e)})

# Save MLP results
df_mlp = pd.DataFrame(results_mlp)
df_mlp = df_mlp.sort_values(by='test_acc', ascending=False)
df_mlp.to_csv('results_MLP_enhanced.csv', index=False)
print(f"\n✓ MLP results saved to 'results_MLP_enhanced.csv'")

# ============================================
# PART 6: FINAL RESULTS & CROSS-VALIDATION
# ============================================
print("\n" + "="*80)
print("PART 6: FINAL RESULTS & VALIDATION")
print("="*80)

# Combine all results
all_results = pd.concat([df_rf_svm, df_mlp], ignore_index=True)
all_results = all_results.sort_values(by='test_acc', ascending=False)
all_results.to_csv('ALL_RESULTS_ENHANCED.csv', index=False)

# Enhanced results analysis
enhanced_results_analysis(all_results)

best_result = all_results.iloc[0]
print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)
print(f"Feature Combination: {best_result['combo']}")
print(f"Model Type: {best_result['model']}")
print(f"PCA Used: {best_result['pca']}")
print(f"CV Accuracy: {best_result['cv_best']:.4f}")
print(f"Test Accuracy: {best_result['test_acc']:.4f}")

# Load best model for further analysis
best_model_path = os.path.join(save_models_dir, f"{best_result['combo']}__{'pca' if best_result['pca'] else 'no_pca'}__{best_result['model']}.pkl")
best_model = joblib.load(best_model_path)

# Feature importance analysis
if best_result['combo'] in feature_names_dict:
    feature_importance = analyze_best_features(best_model, feature_names_dict[best_result['combo']])

# Cross-subject validation
if os.path.exists(test_subject_folder):
    cross_sub_acc = cross_subject_validation(
        best_model, 
        test_subject_folder, 
        priority_combos[best_result['combo']]
    )
    print(f"\n📊 SUMMARY:")
    print(f"  Within-subject accuracy: {best_result['test_acc']:.3f}")
    print(f"  Cross-subject accuracy:  {cross_sub_acc:.3f}")
    print(f"  Generalization drop:     {best_result['test_acc'] - cross_sub_acc:.3f}")

print("\n" + "="*80)
print("✓ ENHANCED PIPELINE COMPLETE!")
print("="*80)
print("\nFiles created:")
print(f"  • X_clean_enhanced.npy - Preprocessed EEG data")
print(f"  • Y_labels_enhanced.npy - Labels")
print(f"  • all_features_enhanced.csv - All features")
print(f"  • results_RF_SVM_enhanced.csv - Random Forest & SVM results")
print(f"  • results_MLP_enhanced.csv - Neural Network results")
print(f"  • ALL_RESULTS_ENHANCED.csv - Combined results")
print(f"  • feature_importance.png - Top feature importances")
print(f"  • {save_models_dir}/ - All trained models")
print("\nBest model:")
print(f"  {best_result['combo']} + {best_result['model']} ({'with' if best_result['pca'] else 'without'} PCA)")
print(f"  Test Accuracy: {best_result['test_acc']:.2%}")
if 'cross_sub_acc' in locals():
    print(f"  Cross-Subject Accuracy: {cross_sub_acc:.2%}")
print("="*80)