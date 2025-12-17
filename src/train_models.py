"""
Moduł do trenowania modeli uczenia maszynowego
z wizualizacjami do pracy dyplomowej
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import learning_curve, cross_val_score

# Opcjonalne biblioteki
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelTrainer:
    """Klasa do trenowania i ewaluacji modeli ML"""

    def __init__(self, data_dir: str = "data/ml_ready"):
        """
        Inicjalizacja trainera

        Args:
            data_dir: Katalog z przygotowanymi danymi ML
        """
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        self.output_dir = Path("data/ml_models")
        self.viz_dir = Path("data/ml_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Wczytuje przygotowane dane ML"""
        logger.info("Wczytywanie danych...")

        self.X_train = pd.read_csv(self.data_dir / "X_train.csv")
        self.X_test = pd.read_csv(self.data_dir / "X_test.csv")
        self.y_train = pd.read_csv(self.data_dir / "y_train.csv").values.ravel()
        self.y_test = pd.read_csv(self.data_dir / "y_test.csv").values.ravel()

        # Wczytaj również skalowane dane
        self.X_train_scaled = pd.read_csv(self.data_dir / "X_train_scaled.csv")
        self.X_test_scaled = pd.read_csv(self.data_dir / "X_test_scaled.csv")

        # Wczytaj scaler
        self.scaler = joblib.load(self.data_dir / "scaler.pkl")

        logger.info(f"X_train shape: {self.X_train.shape}")
        logger.info(f"X_test shape: {self.X_test.shape}")
        logger.info(f"y_train shape: {self.y_train.shape}")
        logger.info(f"y_test shape: {self.y_test.shape}")

    def initialize_models(self):
        """Inicjalizuje modele do trenowania"""
        logger.info("Inicjalizacja modeli...")

        self.models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'use_scaled': True
            },
            'Ridge Regression': {
                'model': Ridge(alpha=1.0, random_state=42),
                'use_scaled': True
            },
            'Lasso Regression': {
                'model': Lasso(alpha=1.0, random_state=42),
                'use_scaled': True
            },
            'Random Forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ),
                'use_scaled': False
            },
        }

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = {
                'model': xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaled': False
            }

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = {
                'model': lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'use_scaled': False
            }

        logger.info(f"Zainicjalizowano {len(self.models)} modeli")

    def train_and_evaluate(self, name: str, model_config: dict):
        """
        Trenuje i ewaluuje pojedynczy model

        Args:
            name: Nazwa modelu
            model_config: Słownik z konfiguracją modelu

        Returns:
            Dict z wynikami
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"TRENOWANIE: {name}")
        logger.info(f"{'='*70}")

        model = model_config['model']
        use_scaled = model_config['use_scaled']

        # Wybierz odpowiednie dane
        if use_scaled:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        else:
            X_train = self.X_train
            X_test = self.X_test

        # Trenuj model
        model.fit(X_train, self.y_train)

        # Predykcje
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metryki - zbiór treningowy
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_r2 = r2_score(self.y_train, y_train_pred)

        # Metryki - zbiór testowy
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(self.y_test, y_test_pred) * 100

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, self.y_train, cv=5,
                                     scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        logger.info(f"\nWyniki na zbiorze TRENINGOWYM:")
        logger.info(f"  MAE:  {train_mae:.2f} kcal")
        logger.info(f"  RMSE: {train_rmse:.2f} kcal")
        logger.info(f"  R²:   {train_r2:.4f}")

        logger.info(f"\nWyniki na zbiorze TESTOWYM:")
        logger.info(f"  MAE:  {test_mae:.2f} kcal")
        logger.info(f"  RMSE: {test_rmse:.2f} kcal")
        logger.info(f"  MAPE: {test_mape:.2f}%")
        logger.info(f"  R²:   {test_r2:.4f}")

        logger.info(f"\nCross-Validation (5-fold):")
        logger.info(f"  MAE: {cv_mae:.2f} ± {cv_std:.2f} kcal")

        results = {
            'name': name,
            'model': model,
            'use_scaled': use_scaled,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_mape': test_mape,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'y_train': self.y_train,
            'y_train_pred': y_train_pred,
            'y_test': self.y_test,
            'y_test_pred': y_test_pred,
        }

        return results

    def train_all_models(self):
        """Trenuje wszystkie modele"""
        logger.info("\n" + "="*70)
        logger.info("ROZPOCZYNAM TRENOWANIE WSZYSTKICH MODELI")
        logger.info("="*70)

        for name, config in self.models.items():
            results = self.train_and_evaluate(name, config)
            self.results[name] = results

        logger.info("\n✓ Wszystkie modele wytrenowane!")

    def save_models(self):
        """Zapisuje wytrenowane modele"""
        logger.info("\nZapisywanie modeli...")

        for name, results in self.results.items():
            filename = name.replace(' ', '_').lower()
            model_path = self.output_dir / f"{filename}.pkl"

            # Zapisz model wraz z informacją o skalowaniu
            model_data = {
                'model': results['model'],
                'use_scaled': results['use_scaled'],
                'scaler': self.scaler if results['use_scaled'] else None,
                'feature_names': self.X_train.columns.tolist(),
                'metrics': {
                    'test_mae': results['test_mae'],
                    'test_rmse': results['test_rmse'],
                    'test_r2': results['test_r2'],
                    'test_mape': results['test_mape'],
                }
            }

            joblib.dump(model_data, model_path)
            logger.info(f"  ✓ {name} -> {model_path}")

        logger.info("✓ Wszystkie modele zapisane!")

    def create_comparison_table(self):
        """Tworzy tabelę porównawczą modeli"""
        logger.info("\n" + "="*90)
        logger.info("PORÓWNANIE MODELI")
        logger.info("="*90)

        header = f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'R²':>10} {'CV MAE':>12}"
        logger.info(header)
        logger.info("-"*90)

        for name, results in self.results.items():
            row = (f"{name:<20} "
                   f"{results['test_mae']:>10.2f} "
                   f"{results['test_rmse']:>10.2f} "
                   f"{results['test_mape']:>9.2f}% "
                   f"{results['test_r2']:>10.4f} "
                   f"{results['cv_mae']:>9.2f}±{results['cv_std']:.1f}")
            logger.info(row)

        logger.info("="*90)

        # Najlepszy model
        best_model = min(self.results.items(), key=lambda x: x[1]['test_mae'])
        logger.info(f"\n🏆 NAJLEPSZY MODEL (według MAE): {best_model[0]}")
        logger.info(f"   MAE = {best_model[1]['test_mae']:.2f} kcal")
        logger.info(f"   R² = {best_model[1]['test_r2']:.4f}\n")

        # Zapisz do pliku
        with open(self.output_dir / "model_comparison.txt", 'w') as f:
            f.write("PORÓWNANIE MODELI\n")
            f.write("="*90 + "\n")
            f.write(header + "\n")
            f.write("-"*90 + "\n")
            for name, results in self.results.items():
                row = (f"{name:<20} "
                       f"{results['test_mae']:>10.2f} "
                       f"{results['test_rmse']:>10.2f} "
                       f"{results['test_mape']:>9.2f}% "
                       f"{results['test_r2']:>10.4f} "
                       f"{results['cv_mae']:>9.2f}±{results['cv_std']:.1f}\n")
                f.write(row)
            f.write("="*90 + "\n\n")
            f.write(f"NAJLEPSZY MODEL: {best_model[0]}\n")
            f.write(f"MAE = {best_model[1]['test_mae']:.2f} kcal\n")
            f.write(f"R² = {best_model[1]['test_r2']:.4f}\n")

    # === WIZUALIZACJE DO PRACY DYPLOMOWEJ ===

    def plot_predictions_comparison(self):
        """Wykres porównujący predykcje wszystkich modeli"""
        logger.info("Tworzę wykres porównawczy predykcji...")

        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]

            y_test = results['y_test']
            y_pred = results['y_test_pred']

            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.5, s=30)

            # Linia idealna
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

            ax.set_xlabel('Rzeczywiste kalorie [kcal]', fontsize=11)
            ax.set_ylabel('Przewidywane kalorie [kcal]', fontsize=11)
            ax.set_title(f"{name}\nR² = {results['test_r2']:.4f}, MAE = {results['test_mae']:.1f} kcal",
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Usuń puste subploty
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        output_file = self.viz_dir / "01_predictions_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Zapisano: {output_file}")

    def plot_residuals(self):
        """Wykresy residuals (błędów predykcji)"""
        logger.info("Tworzę wykresy residuals...")

        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (name, results) in enumerate(self.results.items()):
            ax = axes[idx]

            y_pred = results['y_test_pred']
            residuals = results['y_test'] - y_pred

            ax.scatter(y_pred, residuals, alpha=0.5, s=30)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)

            ax.set_xlabel('Przewidywane kalorie [kcal]', fontsize=11)
            ax.set_ylabel('Residuals (rzeczywiste - przewidywane)', fontsize=11)
            ax.set_title(f"{name}\nMAE = {results['test_mae']:.1f} kcal",
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Usuń puste subploty
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        output_file = self.viz_dir / "02_residuals_plot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Zapisano: {output_file}")

    def plot_feature_importance(self):
        """Wykres istotności cech dla modeli drzewiastych"""
        logger.info("Tworzę wykresy istotności cech...")

        tree_models = {name: res for name, res in self.results.items()
                      if hasattr(res['model'], 'feature_importances_')}

        if not tree_models:
            logger.warning("  Brak modeli z feature_importances_")
            return

        n_models = len(tree_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 8))
        if n_models == 1:
            axes = [axes]

        for idx, (name, results) in enumerate(tree_models.items()):
            ax = axes[idx]

            importances = results['model'].feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15

            feature_names = self.X_train.columns.tolist()

            ax.barh(range(len(indices)), importances[indices], color='steelblue')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Istotność cechy', fontsize=11)
            ax.set_title(f'Top 15 cech - {name}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_file = self.viz_dir / "03_feature_importance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Zapisano: {output_file}")

    def plot_learning_curves(self):
        """Wykresy krzywych uczenia"""
        logger.info("Tworzę krzywe uczenia...")

        # Wybierz 3-4 najlepsze modele
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['test_mae'])[:4]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (name, results) in enumerate(sorted_models):
            if idx >= 4:
                break

            ax = axes[idx]

            model = results['model']
            use_scaled = results['use_scaled']

            X = self.X_train_scaled if use_scaled else self.X_train

            train_sizes, train_scores, val_scores = learning_curve(
                model, X, self.y_train,
                cv=5,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_absolute_error'
            )

            train_mean = -train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = -val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)

            ax.plot(train_sizes, train_mean, label='Training MAE', marker='o')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

            ax.plot(train_sizes, val_mean, label='Validation MAE', marker='o')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

            ax.set_xlabel('Liczba próbek treningowych', fontsize=11)
            ax.set_ylabel('MAE [kcal]', fontsize=11)
            ax.set_title(f'Krzywa uczenia - {name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.viz_dir / "04_learning_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Zapisano: {output_file}")

    def plot_linear_regression_analysis(self):
        """Szczegółowa analiza regresji liniowej"""
        logger.info("Tworzę analizę regresji liniowej...")

        if 'Linear Regression' not in self.results:
            logger.warning("  Brak modelu Linear Regression")
            return

        results = self.results['Linear Regression']
        model = results['model']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Predykcje vs rzeczywiste
        ax = axes[0, 0]
        ax.scatter(results['y_test'], results['y_test_pred'], alpha=0.5)
        min_val = min(results['y_test'].min(), results['y_test_pred'].min())
        max_val = max(results['y_test'].max(), results['y_test_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_xlabel('Rzeczywiste kalorie [kcal]')
        ax.set_ylabel('Przewidywane kalorie [kcal]')
        ax.set_title('Predykcje vs Rzeczywiste wartości')
        ax.grid(True, alpha=0.3)

        # 2. Residuals vs fitted
        ax = axes[0, 1]
        residuals = results['y_test'] - results['y_test_pred']
        ax.scatter(results['y_test_pred'], residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Przewidywane kalorie [kcal]')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted Values')
        ax.grid(True, alpha=0.3)

        # 3. Histogram residuals
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Residuals [kcal]')
        ax.set_ylabel('Częstość')
        ax.set_title('Rozkład residuals')
        ax.grid(True, alpha=0.3)

        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (normalność residuals)')
        ax.grid(True, alpha=0.3)

        plt.suptitle('Analiza Regresji Liniowej', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.viz_dir / "05_linear_regression_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Zapisano: {output_file}")

    def plot_metrics_comparison(self):
        """Wykres słupkowy porównujący metryki"""
        logger.info("Tworzę wykres porównawczy metryk...")

        models = list(self.results.keys())
        mae_scores = [self.results[m]['test_mae'] for m in models]
        rmse_scores = [self.results[m]['test_rmse'] for m in models]
        r2_scores = [self.results[m]['test_r2'] for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # MAE
        axes[0].barh(models, mae_scores, color='steelblue')
        axes[0].set_xlabel('MAE [kcal]')
        axes[0].set_title('Mean Absolute Error', fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # RMSE
        axes[1].barh(models, rmse_scores, color='coral')
        axes[1].set_xlabel('RMSE [kcal]')
        axes[1].set_title('Root Mean Squared Error', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # R²
        axes[2].barh(models, r2_scores, color='mediumseagreen')
        axes[2].set_xlabel('R²')
        axes[2].set_title('R² Score', fontweight='bold')
        axes[2].set_xlim(0, 1)
        axes[2].grid(True, alpha=0.3, axis='x')

        plt.suptitle('Porównanie Metryk Modeli', fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.viz_dir / "06_metrics_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Zapisano: {output_file}")

    def create_all_visualizations(self):
        """Tworzy wszystkie wizualizacje"""
        logger.info("\n" + "="*70)
        logger.info("TWORZENIE WIZUALIZACJI DO PRACY DYPLOMOWEJ")
        logger.info("="*70 + "\n")

        self.plot_predictions_comparison()
        self.plot_residuals()
        self.plot_feature_importance()
        self.plot_learning_curves()
        self.plot_linear_regression_analysis()
        self.plot_metrics_comparison()

        logger.info("\n✓ Wszystkie wizualizacje utworzone!")
        logger.info(f"✓ Zapisane w: {self.viz_dir}")


def main():
    """Główna funkcja trenowania modeli"""
    logger.info("\n" + "="*70)
    logger.info("TRENOWANIE MODELI ML - PREDYKCJA SPALONYCH KALORII")
    logger.info("="*70 + "\n")

    try:
        trainer = ModelTrainer()

        # Wczytaj dane
        trainer.load_data()

        # Inicjalizuj modele
        trainer.initialize_models()

        # Trenuj wszystkie modele
        trainer.train_all_models()

        # Porównaj modele
        trainer.create_comparison_table()

        # Zapisz modele
        trainer.save_models()

        # Utwórz wizualizacje
        trainer.create_all_visualizations()

        logger.info("\n" + "="*70)
        logger.info("✓✓✓ TRENOWANIE ZAKOŃCZONE POMYŚLNIE! ✓✓✓")
        logger.info("="*70 + "\n")

        logger.info("Pliki wyjściowe:")
        logger.info(f"  Modele: {trainer.output_dir}")
        logger.info(f"  Wizualizacje: {trainer.viz_dir}")

    except Exception as e:
        logger.error(f"Wystąpił błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
