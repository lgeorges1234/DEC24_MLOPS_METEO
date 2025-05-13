"""
Module qui crée un mock complet de MLflow pour les tests.
Ce module doit être importé avant tout autre module qui utilise MLflow.
"""
import sys
import pickle
import os
from unittest.mock import MagicMock

# Rendre les MagicMock compatibles avec pickle
MagicMock.__reduce__ = lambda self: (MagicMock, ())

# Classe de base pour tous les objets MLflow
class MockBase:
    def __repr__(self):
        return f"MockMLflow({self.__class__.__name__})"

# Mock pour le module mlflow.tracking
class MockTracking(MockBase):
    class MlflowClient(MockBase):
        def __init__(self, *args, **kwargs):
            self.mock_runs = {}
            self.mock_experiments = {
                "weather_prediction": MagicMock(experiment_id="1")
            }
            self.mock_model_versions = [
                MagicMock(
                    name="test_model",
                    version="1",
                    current_stage="Production",
                    run_id="test_run_id"
                )
            ]
        def get_model_version_by_alias(self, name, alias):
            """Get a model version by alias"""
            # For testing, just return a mock model version
            return MagicMock(
                name=name,
                version="1",
                current_stage="Production",
                run_id="test_run_id"
            )    
            
        def get_experiment_by_name(self, name):
            return self.mock_experiments.get(name)
            
        def create_experiment(self, name, **kwargs):
            exp = MagicMock(experiment_id=str(len(self.mock_experiments) + 1))
            self.mock_experiments[name] = exp
            return exp.experiment_id
            
        def get_run(self, run_id):
            if run_id not in self.mock_runs:
                self.mock_runs[run_id] = MagicMock(
                    info=MagicMock(run_id=run_id),
                    data=MagicMock(tags={}, params={}, metrics={})
                )
            return self.mock_runs[run_id]
            
        def create_run(self, experiment_id, **kwargs):
            run_id = f"test_run_{len(self.mock_runs)}"
            run = MagicMock(
                info=MagicMock(run_id=run_id),
                data=MagicMock(tags={}, params={}, metrics={})
            )
            self.mock_runs[run_id] = run
            return run
            
        def log_param(self, run_id, key, value):
            self.get_run(run_id).data.params[key] = value
            
        def log_metric(self, run_id, key, value, **kwargs):
            self.get_run(run_id).data.metrics[key] = value
            
        def set_tag(self, run_id, key, value):
            self.get_run(run_id).data.tags[key] = value
            
        def get_latest_versions(self, name, **kwargs):
            return self.mock_model_versions
        
        def search_model_versions(self, filter_string="", max_results=None, **kwargs):
            """Recherche les versions de modèles"""
            return self.mock_model_versions
        
        def search_runs(self, experiment_ids=None, filter_string=None, run_view_type=None, max_results=None, **kwargs):
            """Recherche les runs dans un ou plusieurs expériments"""
            mock_runs = [MagicMock(
                info=MagicMock(run_id="test_deployment_run"),
                data=MagicMock(
                    tags={
                        "mlflow.runName": "deployment_run",
                        "mlflow.user": "test_user",
                        "model_version": "1"
                    },
                    params={},
                    metrics={}
                )
            )]
            return mock_runs
            
        def get_model_version_by_alias(self, name, alias):
            """Obtient une version du modèle par son alias"""
            # Pour les tests, on retourne toujours une version valide
            return MagicMock(
                name=name,
                version="1",
                current_stage="Production",
                run_id="test_run_id"
            )
            
        def set_registered_model_alias(self, name, alias, version):
            """Définit un alias pour une version du modèle"""
            # Pas besoin d'implémentation spécifique pour les tests
            pass
    
    # Request_header module
    class request_header:
        @staticmethod
        def get_request_header():
            return {}

# Mock pour le module mlflow.models
class MockModels(MockBase):
    class signature:
        @staticmethod
        def infer_signature(inputs=None, outputs=None, **kwargs):
            return MagicMock()
    
    def save_model(self, *args, **kwargs):
        return "/app/api/data/models/mock_model.pkl"
    
    def load_model(self, *args, **kwargs):
        model = MagicMock()
        model.predict = lambda x: [0]
        return model

# Mock pour le module mlflow.sklearn
class MockSklearn(MockBase):
    def log_model(self, *args, **kwargs):
        return MagicMock(
            model_uri="mock://model/uri",
            model_path="/app/api/data/models/mock_model.pkl"
        )
    
    def load_model(self, *args, **kwargs):
        model = MagicMock()
        model.predict = lambda x: [0]
        model.predict_proba = lambda x: [[0.25, 0.75]]
        return model

# Mock pour le module mlflow.pyfunc
class MockPyfunc(MockBase):
    def load_model(self, *args, **kwargs):
        model = MagicMock()
        model.predict = lambda x: [0]
        return model

# Mock pour un run MLflow actif
class MockRun(MockBase):
    def __init__(self, run_id="test_run_id"):
        self.info = MagicMock(run_id=run_id)
        self.data = MagicMock(tags={}, params={}, metrics={})

# Mock pour le module MLflow principal
class MockMLflow(MockBase):
    def __init__(self):
        self.tracking = MockTracking()
        self.models = MockModels()
        self.sklearn = MockSklearn()
        self.pyfunc = MockPyfunc()
        self.active_run_obj = None
        self._tracking_uri = "http://localhost:5000"
        self._registry_uri = "http://localhost:5000"
        self._current_experiment_id = "1"
        
        # Configure default environment variables for testing
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "1"
        
        # Créer un client par défaut pour faciliter les tests
        self.client = self.tracking.MlflowClient()
        self.mock_experiments = {
            "weather_prediction": MagicMock(experiment_id="1")
        }
        
    # Ajout des méthodes appelées directement sur mlflow
    def get_experiment_by_name(self, name):
        """Méthode appelée directement sur l'objet mlflow"""
        return self.mock_experiments.get(name)

    def create_experiment(self, name, **kwargs):
        """Méthode appelée directement sur l'objet mlflow"""
        exp = MagicMock(experiment_id=str(len(self.mock_experiments) + 1))
        self.mock_experiments[name] = exp
        return exp.experiment_id
    
    def set_experiment(self, experiment_name):
        """Définit l'expérience courante - utilisée directement sur mlflow"""
        if experiment_name not in self.mock_experiments:
            self.mock_experiments[experiment_name] = MagicMock(
                experiment_id=str(len(self.mock_experiments) + 1)
            )
        self._current_experiment_id = self.mock_experiments[experiment_name].experiment_id
        return self.mock_experiments[experiment_name]
        
    def set_tracking_uri(self, uri):
        self._tracking_uri = uri
        
    def get_tracking_uri(self):
        return self._tracking_uri
        
    def set_registry_uri(self, uri):
        self._registry_uri = uri
        
    def get_registry_uri(self):
        return self._registry_uri
        
    def active_run(self):
        return self.active_run_obj
        
    def start_run(self, run_id=None, nested=False, **kwargs):
        class RunContextManager:
            def __init__(self, outer_self, run_id):
                self.outer_self = outer_self
                self.run_id = run_id
                self.run = None
                
            def __enter__(self):
                if self.run_id:
                    self.run = MockRun(self.run_id)
                else:
                    self.run = MockRun()
                self.outer_self.active_run_obj = self.run
                return self.run
                
            def __exit__(self, *args):
                self.outer_self.active_run_obj = None
                
        return RunContextManager(self, run_id)
        
    def end_run(self):
        self.active_run_obj = None
        
    def log_param(self, key, value):
        pass
    
    def log_metric(self, key, value):
        pass
    
    def log_params(self, params):
        pass
    
    def log_metrics(self, metrics):
        pass
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log un artefact - nouvelle méthode requise"""
        pass
    
    def log_artifacts(self, local_dir, artifact_path=None):
        pass
    
    def set_tag(self, key, value):
        pass
    
    def register_model(self, *args, **kwargs):
        return MagicMock(name="test_model", version="1")
    
    def load_model(self, *args, **kwargs):
        model = MagicMock()
        model.predict = lambda x: [0]
        model.predict_proba = lambda x: [[0.25, 0.75]]
        return model
    
    def search_experiments(self, *args, **kwargs):
        return list(self.mock_experiments.values())
    
    # Ajoutons également la méthode search_runs
    def search_runs(self, experiment_ids=None, filter_string=None, max_results=None, **kwargs):
        """Recherche des runs d'expérience"""
        return self.client.search_runs(experiment_ids, filter_string, max_results=max_results, **kwargs)

# Create instances
mock_mlflow = MockMLflow()
mock_mlflow_tracking = mock_mlflow.tracking
mock_mlflow_models = mock_mlflow.models
mock_mlflow_sklearn = mock_mlflow.sklearn
mock_mlflow_pyfunc = mock_mlflow.pyfunc

# Replace real modules with mocks in sys.modules
sys.modules['mlflow'] = mock_mlflow
sys.modules['mlflow.tracking'] = mock_mlflow_tracking
sys.modules['mlflow.models'] = mock_mlflow_models
sys.modules['mlflow.models.signature'] = mock_mlflow_models.signature
sys.modules['mlflow.sklearn'] = mock_mlflow_sklearn
sys.modules['mlflow.pyfunc'] = mock_mlflow_pyfunc