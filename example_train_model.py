#!/usr/bin/env python3
"""
Przykładowy skrypt do treningu modeli ML
Ten skrypt pokazuje jak użyć przygotowanych danych do treningu modeli
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Wczytuje przygotowane dane ML"""
    logger.info("Wczytuję dane...")

    X_train = pd.read_csv("data/ml_ready/X_train.csv")
    X_test = pd.read_csv("data/ml_ready/X_test.csv")
    y_train = pd.read_csv("data/ml_ready/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/ml_ready/y_test.csv").values.ravel()

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Trenuje i ocenia model

    Returns:
        Dict ze statystykami
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRENOWANIE MODELU: {name}")
    logger.info(f"{'='*60}")

    # Trenuj model
    model.fit(X_train, y_train)

    # Predykcja
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metryki dla zbioru treningowego
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Metryki dla zbioru testowego
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    logger.info("\nWyniki na zbiorze TRENINGOWYM:")
    logger.info(f"  MAE:  {train_mae:.2f} kcal")
    logger.info(f"  RMSE: {train_rmse:.2f} kcal")
    logger.info(f"  R²:   {train_r2:.4f}")

    logger.info("\nWyniki na zbiorze TESTOWYM:")
    logger.info(f"  MAE:  {test_mae:.2f} kcal")
    logger.info(f"  RMSE: {test_rmse:.2f} kcal")
    logger.info(f"  MAPE: {test_mape:.2f}%")
    logger.info(f"  R²:   {test_r2:.4f}")

    results = {
        'name': name,
        'model': model,
        'y_test': y_test,
        'y_pred': y_test_pred,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'test_r2': test_r2
    }

    return results


def plot_predictions(results_list, output_dir="data/visualizations"):
    """Tworzy wykres porównujący predykcje różnych modeli"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for i, results in enumerate(results_list):
        ax = axes[i]

        y_test = results['y_test']
        y_pred = results['y_pred']

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.5)

        # Idealna linia predykcji
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Idealna predykcja')

        ax.set_xlabel('Rzeczywiste kalorie [kcal]')
        ax.set_ylabel('Przewidywane kalorie [kcal]')
        ax.set_title(f"{results['name']}\nR² = {results['test_r2']:.4f}, MAE = {results['test_mae']:.1f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_path / "model_predictions_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Zapisano wykres porównawczy: {output_file}")


def plot_feature_importance(model, feature_names, model_name, output_dir="data/visualizations"):
    """Tworzy wykres istotności cech dla modelu drzewiastego"""

    if not hasattr(model, 'feature_importances_'):
        logger.warning(f"Model {model_name} nie ma feature_importances_")
        return

    output_path = Path(output_dir)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15 cech

    plt.figure(figsize=(12, 8))
    plt.title(f'Top 15 najważniejszych cech - {model_name}')
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Istotność cechy')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_file = output_path / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Zapisano wykres istotności cech: {output_file}")


def create_metrics_table(results_list):
    """Tworzy tabelę porównawczą modeli"""

    logger.info("\n" + "="*80)
    logger.info("PORÓWNANIE MODELI")
    logger.info("="*80)

    # Nagłówek
    header = f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10}"
    logger.info(header)
    logger.info("-"*80)

    # Wyniki każdego modelu
    for results in results_list:
        row = f"{results['name']:<25} {results['test_mae']:>10.2f} {results['test_rmse']:>10.2f} {results['test_mape']:>9.2f}% {results['test_r2']:>10.4f}"
        logger.info(row)

    logger.info("="*80 + "\n")

    # Najlepszy model
    best_model = min(results_list, key=lambda x: x['test_mae'])
    logger.info(f"🏆 NAJLEPSZY MODEL (według MAE): {best_model['name']}")
    logger.info(f"   MAE = {best_model['test_mae']:.2f} kcal")
    logger.info(f"   R² = {best_model['test_r2']:.4f}\n")


def main():
    """Główna funkcja"""

    logger.info("\n" + "="*80)
    logger.info("TRENOWANIE MODELI ML - PREDYKCJA SPALONYCH KALORII")
    logger.info("="*80 + "\n")

    try:
        # Wczytaj dane
        X_train, X_test, y_train, y_test = load_data()
        feature_names = X_train.columns.tolist()

        # Definiuj modele do przetestowania
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]

        # Trenuj i oceniaj modele
        results_list = []

        for name, model in models:
            results = evaluate_model(name, model, X_train, X_test, y_train, y_test)
            results_list.append(results)

            # Dla modeli drzewiastych: wykres istotności cech
            if hasattr(model, 'feature_importances_'):
                plot_feature_importance(model, feature_names, name)

        # Porównaj modele
        create_metrics_table(results_list)

        # Wykres porównawczy predykcji
        plot_predictions(results_list)

        logger.info("="*80)
        logger.info("✓✓✓ TRENOWANIE MODELI ZAKOŃCZONE!")
        logger.info("="*80 + "\n")

        logger.info("Pliki wyjściowe:")
        logger.info("  - data/visualizations/model_predictions_comparison.png")
        logger.info("  - data/visualizations/feature_importance_*.png")

    except FileNotFoundError as e:
        logger.error(f"Błąd: Nie znaleziono plików z danymi ML")
        logger.error("Uruchom najpierw: python main.py --step 4")
    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    main()
