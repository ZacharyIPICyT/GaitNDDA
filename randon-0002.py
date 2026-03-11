import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
from sklearn.inspection import permutation_importance
from sklearn.multiclass import OneVsRestClassifier

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CARGA DE DATOS DE MÚLTIPLES ARCHIVOS
# ============================================

print("=" * 70)
print("CARGA DE DATOS - 4 GRUPOS")
print("=" * 70)

# Configuración de archivos
archivos = {
    'ALS': 'zancadas_als.csv',
    'Control': 'zancadas_control.csv',
    'Huntington': 'zancadas_hunt.csv',
    'Parkinson': 'zancadas_park.csv'
}

# Verificar qué archivos existen realmente
print("Verificando archivos disponibles:")
dataframes = []
grupos = []
for nombre, archivo in archivos.items():
    if os.path.exists(archivo):
        print(f"  ✓ {archivo} - {nombre}")
        df_temp = pd.read_csv(archivo)
        df_temp['grupo'] = nombre 
        dataframes.append(df_temp)
        grupos.append(nombre)
    else:
        print(f"  ✗ {archivo} - NO ENCONTRADO")

if len(dataframes) == 0:
    raise FileNotFoundError("No se encontró ningún archivo de datos")

# Combinar todos los dataframes
df = pd.concat(dataframes, ignore_index=True)

print(f"\nTotal de muestras combinadas: {len(df)}")
print(f"Grupos cargados: {', '.join(grupos)}")

# Mostrar distribución por grupo
print("\nDistribución de grupos:")
distribucion = df['grupo'].value_counts()
for grupo, count in distribucion.items():
    print(f"  {grupo}: {count} muestras ({count/len(df)*100:.1f}%)")

# ============================================
# 2. PREPROCESAMIENTO
# ============================================

print("\n" + "=" * 70)
print("PREPROCESAMIENTO")
print("=" * 70)

# Identificar columnas de características (todas las que empiezan con 'p_')
feature_cols = [col for col in df.columns if col.startswith('p_')]
print(f"Número de características temporales: {len(feature_cols)}")

# Verificar si hay valores nulos
print(f"Valores nulos en características: {df[feature_cols].isnull().sum().sum()}")

# Crear matriz de características X y vector objetivo y
X = df[feature_cols].values
y = df['grupo'].values

print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Codificar la variable objetivo
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nClases (codificadas):")
for i, clase in enumerate(le.classes_):
    print(f"  {i}: {clase}")

# Estandarizar características (opcional para Random Forest, pero útil para interpretación)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nCaracterísticas estandarizadas (media=0, std=1)")

# ============================================
# 3. DIVISIÓN EN ENTRENAMIENTO Y PRUEBA
# ============================================

print("\n" + "=" * 70)
print("DIVISIÓN TRAIN/TEST ")
print("=" * 70)

random_state = 42
np.random.seed(random_state)
random.seed(random_state)


test_size = 0.2
train_size = 0.8

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, 
    test_size=test_size, 
    train_size=train_size,
    random_state=random_state, 
    stratify=y_encoded, 
    shuffle=True 
)

print(f"Train size: {X_train.shape[0]} muestras ({train_size*100:.0f}%)")
print(f"Test size: {X_test.shape[0]} muestras ({test_size*100:.0f}%)")

print("\nDistribución en entrenamiento:")
for i, clase in enumerate(le.classes_):
    count = np.sum(y_train == i)
    print(f"  {clase}: {count} ({count/len(y_train)*100:.1f}%)")

print("\nDistribución en prueba:")
for i, clase in enumerate(le.classes_):
    count = np.sum(y_test == i)
    print(f"  {clase}: {count} ({count/len(y_test)*100:.1f}%)")

# ============================================
# 4. ENTRENAMIENTO DE RANDOM FOREST BASE
# ============================================

print("\n" + "=" * 70)
print("ENTRENAMIENTO DE RANDOM FOREST BASE")
print("=" * 70)

# Modelo base
rf_base = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=random_state,
    n_jobs=-1,
    class_weight='balanced'  # Para manejar desbalanceo de clases
)

rf_base.fit(X_train, y_train)

# Predicciones base
y_pred_base_test = rf_base.predict(X_test)
y_pred_base_train = rf_base.predict(X_train)
accuracy_base_test = accuracy_score(y_test, y_pred_base_test)
accuracy_base_train = accuracy_score(y_train, y_pred_base_train)

print(f"Accuracy base (train): {accuracy_base_train:.4f}")
print(f"Accuracy base (test): {accuracy_base_test:.4f}")

# ============================================
# 5. OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================

print("\n" + "=" * 70)
print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
print("=" * 70)

# Grid de hiperparámetros a probar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("Buscando mejores hiperparámetros (esto puede tomar varios minutos)...")
print(f"Total de combinaciones: {np.prod([len(v) for v in param_grid.values()])}")

# Grid search con validación cruzada
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced'),
    param_grid,
    cv=3,  # 3-fold cross-validation para velocidad
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Ejecutar grid search
grid_search.fit(X_train, y_train)

print(f"\nMejores parámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Mejor accuracy CV: {grid_search.best_score_:.4f}")

# Modelo optimizado
rf_optimized = grid_search.best_estimator_

# ============================================
# 6. EVALUACIÓN DEL MODELO OPTIMIZADO
# ============================================

print("\n" + "=" * 70)
print("EVALUACIÓN DEL MODELO OPTIMIZADO")
print("=" * 70)

# Predicciones en train y test
y_pred_train = rf_optimized.predict(X_train)
y_pred_test = rf_optimized.predict(X_test)
y_prob_train = rf_optimized.predict_proba(X_train)
y_prob_test = rf_optimized.predict_proba(X_test)

# Métricas principales
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Accuracy en entrenamiento: {accuracy_train:.4f}")
print(f"Accuracy en prueba: {accuracy_test:.4f}")
print(f"Mejora vs modelo base (test): {accuracy_test - accuracy_base_test:.4f}")

# Verificar sobreajuste
if accuracy_train - accuracy_test > 0.1:
    print("\n POSIBLE SOBREAJUSTE: La diferencia train-test es > 10%")
else:
    print("\n Buen balance: La diferencia train-test es aceptable")

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT (TEST)")
print("=" * 50)
print(classification_report(y_test, y_pred_test, target_names=le.classes_))

# Matrices de confusión
cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

# Normalizar matrices
cm_train_norm = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
cm_test_norm = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]

# ============================================
# 7. VISUALIZACIONES
# ============================================

print("\n" + "=" * 70)
print("VISUALIZACIONES")
print("=" * 70)

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 15))

# 7.1 Matriz de confusión - ENTRENAMIENTO
ax1 = plt.subplot(3, 3, 1)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax1.set_title(f'Matriz de Confusión - ENTRENAMIENTO\n(Accuracy: {accuracy_train:.4f})', fontsize=14)
ax1.set_xlabel('Predicho')
ax1.set_ylabel('Real')

# 7.2 Matriz de confusión normalizada - ENTRENAMIENTO
ax2 = plt.subplot(3, 3, 2)
sns.heatmap(cm_train_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax2.set_title('Matriz de Confusión Normalizada - ENTRENAMIENTO', fontsize=14)
ax2.set_xlabel('Predicho')
ax2.set_ylabel('Real')

# 7.3 Matriz de confusión - PRUEBA
ax3 = plt.subplot(3, 3, 4)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=ax3,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax3.set_title(f'Matriz de Confusión - PRUEBA\n(Accuracy: {accuracy_test:.4f})', fontsize=14)
ax3.set_xlabel('Predicho')
ax3.set_ylabel('Real')

# 7.4 Matriz de confusión normalizada - PRUEBA
ax4 = plt.subplot(3, 3, 5)
sns.heatmap(cm_test_norm, annot=True, fmt='.2f', cmap='Oranges', ax=ax4,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax4.set_title('Matriz de Confusión Normalizada - PRUEBA', fontsize=14)
ax4.set_xlabel('Predicho')
ax4.set_ylabel('Real')

# 7.5 Comparación de accuracy
ax5 = plt.subplot(3, 3, 3)
bars = ax5.bar(['Entrenamiento', 'Prueba'], [accuracy_train, accuracy_test], 
               color=['skyblue', 'orange'])
ax5.set_ylim([0, 1.1])
ax5.set_ylabel('Accuracy')
ax5.set_title('Comparación de Accuracy')
# Añadir valores en las barras
for bar, acc in zip(bars, [accuracy_train, accuracy_test]):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# 7.6 Importancia de características
importances = rf_optimized.feature_importances_
indices = np.argsort(importances)[::-1]

top_n = 30
ax6 = plt.subplot(3, 3, 6)
ax6.barh(range(top_n), importances[indices[:top_n]][::-1])
ax6.set_yticks(range(top_n))
ax6.set_yticklabels([feature_cols[i] for i in indices[:top_n]][::-1], fontsize=8)
ax6.set_title(f'Top {top_n} Características más Importantes', fontsize=14)
ax6.set_xlabel('Importancia')

# 7.7 Distribución de probabilidades por clase (train)
ax7 = plt.subplot(3, 3, 7)
prob_means_train = np.mean(y_prob_train, axis=0)
ax7.bar(le.classes_, prob_means_train, color='skyblue', alpha=0.7)
ax7.set_title('Probabilidad Promedio por Clase - ENTRENAMIENTO', fontsize=12)
ax7.set_xlabel('Clase')
ax7.set_ylabel('Probabilidad promedio')
ax7.tick_params(axis='x', rotation=45)

# 7.8 Distribución de probabilidades por clase (test)
ax8 = plt.subplot(3, 3, 8)
prob_means_test = np.mean(y_prob_test, axis=0)
ax8.bar(le.classes_, prob_means_test, color='orange', alpha=0.7)
ax8.set_title('Probabilidad Promedio por Clase - PRUEBA', fontsize=12)
ax8.set_xlabel('Clase')
ax8.set_ylabel('Probabilidad promedio')
ax8.tick_params(axis='x', rotation=45)

# 7.9 Curvas ROC (One-vs-Rest)
ax9 = plt.subplot(3, 3, 9)
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(le.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    ax9.plot(fpr[i], tpr[i], label=f'{le.classes_[i]} (AUC = {roc_auc[i]:.2f})')

ax9.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
ax9.set_xlim([0.0, 1.0])
ax9.set_ylim([0.0, 1.05])
ax9.set_xlabel('Tasa de Falsos Positivos')
ax9.set_ylabel('Tasa de Verdaderos Positivos')
ax9.set_title('Curvas ROC (One-vs-Rest)', fontsize=14)
ax9.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('random_forest_matrices_confusion_completas.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigura guardada como 'random_forest_matrices_confusion_completas.png'")

# ============================================
# 8. MATRICES DE CONFUSIÓN INDIVIDUALES 
# ============================================

# Guardar matrices individuales por separado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de entrenamiento
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax1.set_title(f'Matriz de Confusión - ENTRENAMIENTO\n(Accuracy: {accuracy_train:.4f})', fontsize=14)
ax1.set_xlabel('Predicho')
ax1.set_ylabel('Real')

# Matriz de prueba
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', ax=ax2,
            xticklabels=le.classes_, yticklabels=le.classes_)
ax2.set_title(f'Matriz de Confusión - PRUEBA\n(Accuracy: {accuracy_test:.4f})', fontsize=14)
ax2.set_xlabel('Predicho')
ax2.set_ylabel('Real')

plt.tight_layout()
plt.savefig('random_forest_matrices_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

print("Figura guardada como 'random_forest_matrices_comparacion.png'")

# ============================================
# 9. VALIDACIÓN CRUZADA
# ============================================

print("\n" + "=" * 70)
print("VALIDACIÓN CRUZADA (5-FOLD)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
cv_scores = cross_val_score(rf_optimized, X_scaled, y_encoded, cv=cv, scoring='accuracy')

print(f"CV Accuracy (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Scores individuales: {[f'{score:.4f}' for score in cv_scores]}")

# ============================================
# 10. ANÁLISIS DE IMPORTANCIA POR PERMUTACIÓN
# ============================================

print("\n" + "=" * 70)
print("IMPORTANCIA POR PERMUTACIÓN")
print("=" * 70)

# Calcular importancia por permutación (más confiable que feature_importances_)
perm_importance = permutation_importance(
    rf_optimized, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1
)

# Mostrar top 10 características por permutación
top_perm_indices = np.argsort(perm_importance.importances_mean)[::-1][:10]
print("\nTop 10 características por importancia de permutación:")
for i, idx in enumerate(top_perm_indices):
    print(f"  {i+1}. {feature_cols[idx]}: {perm_importance.importances_mean[idx]:.6f}")

# ============================================
# 11. ANÁLISIS DE ERRORES
# ============================================

print("\n" + "=" * 70)
print("ANÁLISIS DE ERRORES")
print("=" * 70)

# Encontrar muestras mal clasificadas en test
errores = np.where(y_pred_test != y_test)[0]

if len(errores) > 0:
    print(f"Total de errores en TEST: {len(errores)} de {len(y_test)} ({len(errores)/len(y_test)*100:.1f}%)")
    
    # Mostrar algunos ejemplos de errores
    print("\nEjemplos de muestras mal clasificadas en TEST:")
    for i, idx in enumerate(errores[:10]):
        real = le.classes_[y_test[idx]]
        pred = le.classes_[y_pred_test[idx]]
        prob_real = y_prob_test[idx, y_test[idx]]
        prob_pred = y_prob_test[idx, y_pred_test[idx]]
        print(f"  Muestra {idx}: Real={real}, Pred={pred} "
              f"(Prob real: {prob_real:.3f}, Prob pred: {prob_pred:.3f})")
    
    # Matriz de errores por clase
    print("\nMatriz de errores por clase (TEST):")
    error_matrix = pd.crosstab(
        [le.classes_[i] for i in y_test[errores]],
        [le.classes_[i] for i in y_pred_test[errores]],
        rownames=['Real'], colnames=['Predicho']
    )
    print(error_matrix)
else:
    print("¡No hay errores! Perfecta clasificación en test")

# ============================================
# 12. PREDICCIÓN EN NUEVOS DATOS (EJEMPLO)
# ============================================

print("\n" + "=" * 70)
print("PREDICCIÓN EN NUEVOS DATOS")
print("=" * 70)

# Tomar una muestra aleatoria de cada clase como ejemplo
print("Ejemplo de predicción para una muestra aleatoria de cada clase:")
for clase_idx, clase in enumerate(le.classes_):
    # Encontrar una muestra de esta clase en test
    mask = y_test == clase_idx
    if np.any(mask):
        # Seleccionar una muestra aleatoria de esta clase
        indices_clase = np.where(mask)[0]
        idx_aleatorio = np.random.choice(indices_clase)
        muestra = X_test[idx_aleatorio:idx_aleatorio+1]
        pred = rf_optimized.predict(muestra)[0]
        prob = rf_optimized.predict_proba(muestra)[0]
        
        print(f"\n  Muestra REAL: {clase}")
        print(f"  Predicción: {le.classes_[pred]}")
        print(f"  Probabilidades:")
        for j, prob_clase in enumerate(prob):
            print(f"    {le.classes_[j]}: {prob_clase:.3f}")

# ============================================
# 13. GUARDAR MODELO PARA USO FUTURO
# ============================================

print("\n" + "=" * 70)
print("GUARDAR MODELO")
print("=" * 70)

import joblib

# Guardar modelo y preprocesadores
modelo_completo = {
    'modelo': rf_optimized,
    'label_encoder': le,
    'scaler': scaler,
    'feature_names': feature_cols,
    'train_test_split': {
        'random_state': random_state,
        'test_size': test_size,
        'train_size': train_size
    },
    'accuracy': {
        'train': accuracy_train,
        'test': accuracy_test
    }
}

joblib.dump(modelo_completo, 'modelo_random_forest_4grupos.pkl')
print("Modelo guardado como 'modelo_random_forest_4grupos.pkl'")

# ============================================
# 14. RESUMEN FINAL
# ============================================

print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"Problema: Clasificación de {len(le.classes_)} grupos")
print(f"Grupos: {', '.join(le.classes_)}")
print(f"Total muestras: {len(df)}")
print(f"Características: {len(feature_cols)}")
print(f"División: {train_size*100:.0f}% entrenamiento / {test_size*100:.0f}% prueba")
print(f"Modelo: Random Forest Classifier")
print(f"Número de árboles: {rf_optimized.n_estimators}")
print(f"Profundidad máxima: {rf_optimized.max_depth}")
print(f"\nRESULTADOS:")
print(f"  Accuracy entrenamiento: {accuracy_train:.4f}")
print(f"  Accuracy prueba: {accuracy_test:.4f}")
print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f}")
print(f"  Diferencia train-test: {accuracy_train - accuracy_test:.4f}")
print("=" * 70)