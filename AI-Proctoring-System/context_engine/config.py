"""
Configuration Service - System configuration management.

This module provides the ConfigurationService class that handles
loading, saving, and managing system configuration from various sources.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .models import SystemConfiguration, DetectionType, DeploymentMode
from .interfaces import ConfigurationProvider


class ConfigurationService(ConfigurationProvider):
    """
    Service for managing system configuration.
    
    Handles loading configuration from files, environment variables,
    and provides hot-reload capabilities for runtime updates.
    """
    
    def __init__(self, config_file: str = "config/system_config.json"):
        """
        Initialize configuration service.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.watchers = []
        
        # Ensure config directory exists
        config_dir = os.path.dirname(config_file)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
    
    def load_configuration(self) -> SystemConfiguration:
        """Load system configuration from file and environment variables."""
        try:
            # Start with default configuration
            config_data = self._get_default_config()
            
            # Load from file if it exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config_data.update(file_config)
                    self.logger.info(f"Loaded configuration from {self.config_file}")
            else:
                self.logger.info(f"Config file {self.config_file} not found, using defaults")
                # Create default config file
                self._create_default_config_file()
            
            # Override with environment variables
            config_data = self._apply_environment_overrides(config_data)
            
            # Convert to SystemConfiguration object
            return self._dict_to_configuration(config_data)
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Return default configuration on error
            return SystemConfiguration()
    
    def save_configuration(self, config: SystemConfiguration) -> bool:
        """Save system configuration to file."""
        try:
            config_data = config.to_dict()
            
            # Ensure config directory exists
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def watch_for_changes(self, callback) -> None:
        """Watch for configuration changes and call callback when detected."""
        # Simple implementation - could be enhanced with file system watchers
        self.watchers.append(callback)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            "detection_thresholds": {
                "gaze_away": 0.7,
                "lip_movement": 0.6,
                "suspicious_speech": 0.8,
                "multiple_people": 0.9,
                "mobile_detected": 0.8,
                "face_spoof": 0.9,
                "head_pose_suspicious": 0.6,
                "audio_anomaly": 0.7
            },
            "correlation_window_seconds": 5,
            "max_correlation_events": 10,
            "evidence_retention_days": 30,
            "max_evidence_file_size_mb": 100,
            "deployment_mode": "development",
            "max_processing_latency_ms": 500,
            "enable_gpu_acceleration": True
        }
    
    def _create_default_config_file(self) -> None:
        """Create default configuration file."""
        default_config = self._get_default_config()
        
        # Ensure config directory exists
        config_dir = os.path.dirname(self.config_file)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        self.logger.info(f"Created default configuration file: {self.config_file}")
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Environment variable mappings
        env_mappings = {
            "PROCTORING_CORRELATION_WINDOW": ("correlation_window_seconds", int),
            "PROCTORING_MAX_LATENCY": ("max_processing_latency_ms", int),
            "PROCTORING_EVIDENCE_RETENTION": ("evidence_retention_days", int),
            "PROCTORING_MAX_FILE_SIZE": ("max_evidence_file_size_mb", int),
            "PROCTORING_ENABLE_GPU": ("enable_gpu_acceleration", lambda x: x.lower() == "true"),
            "DEPLOYMENT_MODE": ("deployment_mode", str)
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    config_data[config_key] = converter(env_value)
                    self.logger.info(f"Applied environment override: {config_key} = {config_data[config_key]}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
        
        return config_data
    
    def _dict_to_configuration(self, config_data: Dict[str, Any]) -> SystemConfiguration:
        """Convert configuration dictionary to SystemConfiguration instance."""
        # Convert detection thresholds
        detection_thresholds = {}
        for key, value in config_data.get("detection_thresholds", {}).items():
            try:
                detection_type = DetectionType(key)
                detection_thresholds[detection_type] = float(value)
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid detection threshold: {key}={value}")
        
        # Convert deployment mode
        deployment_mode = DeploymentMode.DEVELOPMENT
        try:
            deployment_mode = DeploymentMode(config_data.get("deployment_mode", "development"))
        except ValueError:
            self.logger.warning(f"Invalid deployment mode: {config_data.get('deployment_mode')}")
        
        return SystemConfiguration(
            detection_thresholds=detection_thresholds,
            correlation_window_seconds=int(config_data.get("correlation_window_seconds", 5)),
            max_correlation_events=int(config_data.get("max_correlation_events", 10)),
            evidence_retention_days=int(config_data.get("evidence_retention_days", 30)),
            max_evidence_file_size_mb=int(config_data.get("max_evidence_file_size_mb", 100)),
            deployment_mode=deployment_mode,
            max_processing_latency_ms=int(config_data.get("max_processing_latency_ms", 500)),
            enable_gpu_acceleration=bool(config_data.get("enable_gpu_acceleration", True))
        )
        Args:
            config_dir: Directory containing configuration files
            enable_hot_reload: Whether to enable automatic configuration reloading
        """
        self._logger = logging.getLogger(__name__)
        self._config_lock = Lock()
        self._config: Optional[SystemConfiguration] = None
        self._config_file_path: Optional[Path] = None
        self._last_modified: Optional[float] = None
        self._enable_hot_reload = enable_hot_reload
        
        # Set up configuration directory
        if config_dir:
            self._config_dir = Path(config_dir)
        else:
            # Default to config directory relative to project root
            project_root = Path(__file__).parent.parent
            self._config_dir = project_root / "config"
        
        self._config_dir.mkdir(exist_ok=True)
        
        # Load initial configuration
        self._load_configuration()
    
    def get_configuration(self) -> SystemConfiguration:
        """
        Get the current system configuration.
        
        Returns:
            Current SystemConfiguration instance
        """
        with self._config_lock:
            if self._enable_hot_reload:
                self._check_for_updates()
            
            if self._config is None:
                raise ConfigurationError("Configuration not loaded")
            
            return self._config
    
    def reload_configuration(self) -> bool:
        """
        Force reload the configuration from file.
        
        Returns:
            True if configuration was successfully reloaded, False otherwise
        """
        try:
            with self._config_lock:
                self._load_configuration()
                self._logger.info("Configuration reloaded successfully")
                return True
        except Exception as e:
            self._logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def update_configuration(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values and save to file.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            with self._config_lock:
                if self._config is None:
                    raise ConfigurationError("Configuration not loaded")
                
                # Create updated configuration dictionary
                current_config = self._config.to_dict()
                current_config.update(updates)
                
                # Validate the updated configuration
                is_valid, errors = validate_system_configuration(current_config)
                if not is_valid:
                    raise ConfigurationError(f"Invalid configuration: {', '.join(errors)}")
                
                # Create new configuration instance
                new_config = self._dict_to_configuration(current_config)
                
                # Save to file
                if self._config_file_path:
                    self._save_configuration_to_file(current_config, self._config_file_path)
                
                # Update in-memory configuration
                self._config = new_config
                self._last_modified = time.time()
                
                self._logger.info("Configuration updated successfully")
                return True
                
        except Exception as e:
            self._logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_deployment_mode(self) -> DeploymentMode:
        """Get the current deployment mode."""
        return self.get_configuration().deployment_mode
    
    def get_detection_threshold(self, detection_type: DetectionType) -> float:
        """Get the confidence threshold for a specific detection type."""
        return self.get_configuration().get_threshold(detection_type)
    
    def apply_sensitivity_profile(self, profile: SensitivityProfile) -> bool:
        """
        Apply a predefined sensitivity profile to the current configuration.
        
        Args:
            profile: The sensitivity profile to apply
            
        Returns:
            True if profile was successfully applied, False otherwise
        """
        try:
            profile_config = self.create_sensitivity_profile(profile)
            return self.update_configuration(profile_config)
        except Exception as e:
            self._logger.error(f"Failed to apply sensitivity profile {profile.value}: {e}")
            return False
    
    def create_sensitivity_profile(self, profile: SensitivityProfile) -> Dict[str, Any]:
        """
        Create configuration settings for a specific sensitivity profile.
        
        Args:
            profile: The sensitivity profile to create
            
        Returns:
            Configuration dictionary with profile-specific settings
        """
        if profile == SensitivityProfile.STRICT:
            return {
                "detection_thresholds": {
                    "gaze_away": 0.9,
                    "lip_movement": 0.8,
                    "suspicious_speech": 0.95,
                    "multiple_people": 0.98,
                    "mobile_detected": 0.95,
                    "face_spoof": 0.98,
                    "head_pose_suspicious": 0.8,
                    "audio_anomaly": 0.9
                },
                "correlation_window_seconds": 3,
                "alert_combination_rules": {
                    "gaze_and_speech": {
                        "detection_types": ["gaze_away", "suspicious_speech"],
                        "weight_multiplier": 2.0,
                        "time_window_seconds": 2,
                        "minimum_confidence": 0.8
                    },
                    "lip_and_audio": {
                        "detection_types": ["lip_movement", "audio_anomaly"],
                        "weight_multiplier": 1.8,
                        "time_window_seconds": 1,
                        "minimum_confidence": 0.7
                    }
                }
            }
        elif profile == SensitivityProfile.BALANCED:
            return {
                "detection_thresholds": {
                    "gaze_away": 0.7,
                    "lip_movement": 0.6,
                    "suspicious_speech": 0.8,
                    "multiple_people": 0.9,
                    "mobile_detected": 0.8,
                    "face_spoof": 0.9,
                    "head_pose_suspicious": 0.6,
                    "audio_anomaly": 0.7
                },
                "correlation_window_seconds": 5,
                "alert_combination_rules": {
                    "gaze_and_speech": {
                        "detection_types": ["gaze_away", "suspicious_speech"],
                        "weight_multiplier": 1.5,
                        "time_window_seconds": 3,
                        "minimum_confidence": 0.6
                    },
                    "lip_and_audio": {
                        "detection_types": ["lip_movement", "audio_anomaly"],
                        "weight_multiplier": 1.3,
                        "time_window_seconds": 2,
                        "minimum_confidence": 0.5
                    }
                }
            }
        elif profile == SensitivityProfile.LENIENT:
            return {
                "detection_thresholds": {
                    "gaze_away": 0.5,
                    "lip_movement": 0.4,
                    "suspicious_speech": 0.6,
                    "multiple_people": 0.7,
                    "mobile_detected": 0.6,
                    "face_spoof": 0.7,
                    "head_pose_suspicious": 0.4,
                    "audio_anomaly": 0.5
                },
                "correlation_window_seconds": 8,
                "alert_combination_rules": {
                    "gaze_and_speech": {
                        "detection_types": ["gaze_away", "suspicious_speech"],
                        "weight_multiplier": 1.2,
                        "time_window_seconds": 5,
                        "minimum_confidence": 0.4
                    },
                    "lip_and_audio": {
                        "detection_types": ["lip_movement", "audio_anomaly"],
                        "weight_multiplier": 1.1,
                        "time_window_seconds": 4,
                        "minimum_confidence": 0.3
                    }
                }
            }
        else:
            raise ValueError(f"Unknown sensitivity profile: {profile}")
    
    def migrate_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate configuration from older versions to current version.
        
        Args:
            config_data: Configuration data to migrate
            
        Returns:
            Migrated configuration data
        """
        current_version = config_data.get("config_version", "0.9")  # Assume old version if not specified
        
        if current_version == self.CURRENT_CONFIG_VERSION:
            return config_data  # No migration needed
        
        self._logger.info(f"Migrating configuration from version {current_version} to {self.CURRENT_CONFIG_VERSION}")
        
        # Migration from version 0.9 to 1.0
        if current_version == "0.9":
            config_data = self._migrate_from_0_9_to_1_0(config_data)
        
        # Add version information
        config_data["config_version"] = self.CURRENT_CONFIG_VERSION
        config_data["migration_timestamp"] = datetime.now().isoformat()
        
        return config_data
    
    def _migrate_from_0_9_to_1_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Migrate configuration from version 0.9 to 1.0.
        
        Changes in 1.0:
        - Added head_pose_suspicious and audio_anomaly detection types
        - Restructured alert_combination_rules format
        - Added max_correlation_events setting
        """
        migrated = config_data.copy()
        
        # Add new detection thresholds if missing
        if "detection_thresholds" in migrated:
            thresholds = migrated["detection_thresholds"]
            if "head_pose_suspicious" not in thresholds:
                thresholds["head_pose_suspicious"] = 0.6
            if "audio_anomaly" not in thresholds:
                thresholds["audio_anomaly"] = 0.7
        
        # Add max_correlation_events if missing
        if "max_correlation_events" not in migrated:
            migrated["max_correlation_events"] = 10
        
        # Migrate old alert rules format if present
        if "alert_rules" in migrated and "alert_combination_rules" not in migrated:
            old_rules = migrated.pop("alert_rules")
            migrated["alert_combination_rules"] = self._convert_old_alert_rules(old_rules)
        
        return migrated
    
    def _convert_old_alert_rules(self, old_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old alert rules format to new combination rules format."""
        new_rules = {}
        
        for rule_name, rule_data in old_rules.items():
            if isinstance(rule_data, dict):
                new_rules[rule_name] = {
                    "detection_types": rule_data.get("types", []),
                    "weight_multiplier": rule_data.get("multiplier", 1.0),
                    "time_window_seconds": rule_data.get("window", 5),
                    "minimum_confidence": rule_data.get("min_confidence", 0.5)
                }
        
        return new_rules
    
    def get_configuration_version(self) -> str:
        """Get the current configuration version."""
        return self.CURRENT_CONFIG_VERSION
    
    def backup_configuration(self) -> Optional[Path]:
        """
        Create a backup of the current configuration file.
        
        Returns:
            Path to the backup file, or None if backup failed
        """
        if not self._config_file_path or not self._config_file_path.exists():
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self._config_file_path.stem}_backup_{timestamp}.json"
            backup_path = self._config_file_path.parent / backup_name
            
            import shutil
            shutil.copy2(self._config_file_path, backup_path)
            
            self._logger.info(f"Configuration backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            self._logger.error(f"Failed to backup configuration: {e}")
            return None

    def create_default_configuration(self, deployment_mode: DeploymentMode) -> Dict[str, Any]:
        """
        Create default configuration template for specified deployment mode.
        
        Args:
            deployment_mode: Target deployment mode
            
        Returns:
            Default configuration dictionary
        """
        base_config = {
            "detection_thresholds": {
                "gaze_away": 0.7,
                "lip_movement": 0.6,
                "suspicious_speech": 0.8,
                "multiple_people": 0.9,
                "mobile_detected": 0.8,
                "face_spoof": 0.9,
                "head_pose_suspicious": 0.6,
                "audio_anomaly": 0.7
            },
            "correlation_window_seconds": 5,
            "max_correlation_events": 10,
            "evidence_retention_days": 30,
            "max_evidence_file_size_mb": 100,
            "deployment_mode": deployment_mode.value,
            "max_processing_latency_ms": 500,
            "enable_gpu_acceleration": True,
            "alert_combination_rules": {
                "gaze_and_speech": {
                    "detection_types": ["gaze_away", "suspicious_speech"],
                    "weight_multiplier": 1.5,
                    "time_window_seconds": 3,
                    "minimum_confidence": 0.6
                },
                "lip_and_audio": {
                    "detection_types": ["lip_movement", "audio_anomaly"],
                    "weight_multiplier": 1.3,
                    "time_window_seconds": 2,
                    "minimum_confidence": 0.5
                },
                "multiple_people_mobile": {
                    "detection_types": ["multiple_people", "mobile_detected"],
                    "weight_multiplier": 2.0,
                    "time_window_seconds": 5,
                    "minimum_confidence": 0.7
                }
            }
        }
        
        # Adjust configuration based on deployment mode
        if deployment_mode == DeploymentMode.DEVELOPMENT:
            base_config["max_processing_latency_ms"] = 1000  # More lenient for development
            base_config["evidence_retention_days"] = 7  # Shorter retention for dev
        elif deployment_mode == DeploymentMode.DOCKER:
            base_config["enable_gpu_acceleration"] = False  # Conservative default for containers
            base_config["max_evidence_file_size_mb"] = 50  # Smaller files for containers
        elif deployment_mode == DeploymentMode.PRODUCTION:
            # Stricter thresholds for production
            base_config["detection_thresholds"]["face_spoof"] = 0.95
            base_config["detection_thresholds"]["multiple_people"] = 0.95
            base_config["max_processing_latency_ms"] = 300
        
        # Add version information
        base_config["config_version"] = self.CURRENT_CONFIG_VERSION
        
        return base_config
    
    def _load_configuration(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Determine configuration file path
            config_file = self._determine_config_file()
            
            # Load base configuration from file
            if config_file.exists():
                config_data = self._load_from_file(config_file)
                
                # Migrate configuration if needed
                config_data = self.migrate_configuration(config_data)
                
                # Save migrated configuration back to file if it was migrated
                if config_data.get("migration_timestamp"):
                    self.backup_configuration()  # Backup before saving migrated version
                    self._save_configuration_to_file(config_data, config_file)
                
                self._config_file_path = config_file
                self._last_modified = config_file.stat().st_mtime
            else:
                # Create default configuration
                deployment_mode = self._detect_deployment_mode()
                config_data = self.create_default_configuration(deployment_mode)
                self._save_configuration_to_file(config_data, config_file)
                self._config_file_path = config_file
                self._last_modified = time.time()
            
            # Apply environment variable overrides
            config_data = self._apply_environment_overrides(config_data)
            
            # Validate configuration
            is_valid, errors = validate_system_configuration(config_data)
            if not is_valid:
                raise ConfigurationError(f"Invalid configuration: {', '.join(errors)}")
            
            # Create configuration instance
            self._config = self._dict_to_configuration(config_data)
            
            self._logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            self._logger.error(f"Failed to load configuration: {e}")
            # Fall back to default configuration
            deployment_mode = self._detect_deployment_mode()
            config_data = self.create_default_configuration(deployment_mode)
            self._config = self._dict_to_configuration(config_data)
    
    def _determine_config_file(self) -> Path:
        """Determine which configuration file to use based on environment."""
        deployment_mode = self._detect_deployment_mode()
        
        # Check for mode-specific configuration files
        mode_specific_file = self._config_dir / f"{deployment_mode.value}.json"
        if mode_specific_file.exists():
            return mode_specific_file
        
        # Fall back to default configuration file
        return self._config_dir / "default.json"
    
    def _detect_deployment_mode(self) -> DeploymentMode:
        """Detect the current deployment mode from environment."""
        # Check environment variable first
        mode_env = os.getenv("DEPLOYMENT_MODE", "").lower()
        if mode_env:
            try:
                return DeploymentMode(mode_env)
            except ValueError:
                pass
        
        # Check for Docker environment
        if os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
            return DeploymentMode.DOCKER
        
        # Check Flask environment
        flask_env = os.getenv("FLASK_ENV", "").lower()
        if flask_env == "production":
            return DeploymentMode.PRODUCTION
        elif flask_env == "development":
            return DeploymentMode.DEVELOPMENT
        
        # Default to development
        return DeploymentMode.DEVELOPMENT
    
    def _load_from_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file {config_file}: {e}")
        except IOError as e:
            raise ConfigurationError(f"Failed to read configuration file {config_file}: {e}")
    
    def _save_configuration_to_file(self, config_data: Dict[str, Any], config_file: Path) -> None:
        """Save configuration to JSON file."""
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            raise ConfigurationError(f"Failed to write configuration file {config_file}: {e}")
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Environment variable mappings
        env_mappings = {
            "PROCTORING_CORRELATION_WINDOW": ("correlation_window_seconds", int),
            "PROCTORING_MAX_LATENCY": ("max_processing_latency_ms", int),
            "PROCTORING_EVIDENCE_RETENTION": ("evidence_retention_days", int),
            "PROCTORING_MAX_FILE_SIZE": ("max_evidence_file_size_mb", int),
            "PROCTORING_ENABLE_GPU": ("enable_gpu_acceleration", lambda x: x.lower() == "true"),
            "DEPLOYMENT_MODE": ("deployment_mode", str)
        }
        
        for env_var, (config_key, converter) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    config_data[config_key] = converter(env_value)
                    self._logger.info(f"Applied environment override: {config_key} = {config_data[config_key]}")
                except (ValueError, TypeError) as e:
                    self._logger.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
        
        # Handle detection threshold overrides
        for detection_type in DetectionType:
            env_var = f"PROCTORING_THRESHOLD_{detection_type.value.upper()}"
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    threshold = float(env_value)
                    if 0.0 <= threshold <= 1.0:
                        config_data["detection_thresholds"][detection_type.value] = threshold
                        self._logger.info(f"Applied threshold override: {detection_type.value} = {threshold}")
                    else:
                        self._logger.warning(f"Invalid threshold value {env_var}={env_value}: must be between 0.0 and 1.0")
                except ValueError:
                    self._logger.warning(f"Invalid threshold value {env_var}={env_value}: not a valid float")
        
        return config_data
    
    def _dict_to_configuration(self, config_data: Dict[str, Any]) -> SystemConfiguration:
        """Convert configuration dictionary to SystemConfiguration instance."""
        # Convert detection thresholds
        detection_thresholds = {}
        for key, value in config_data.get("detection_thresholds", {}).items():
            try:
                detection_type = DetectionType(key)
                detection_thresholds[detection_type] = float(value)
            except (ValueError, TypeError):
                self._logger.warning(f"Invalid detection threshold: {key}={value}")
        
        # Convert deployment mode
        deployment_mode = DeploymentMode.DEVELOPMENT
        try:
            deployment_mode = DeploymentMode(config_data.get("deployment_mode", "development"))
        except ValueError:
            self._logger.warning(f"Invalid deployment mode: {config_data.get('deployment_mode')}")
        
        # Convert combination rules
        combination_rules = {}
        for rule_name, rule_data in config_data.get("alert_combination_rules", {}).items():
            try:
                detection_types = [DetectionType(dt) for dt in rule_data.get("detection_types", [])]
                combination_rules[rule_name] = CombinationRule(
                    detection_types=detection_types,
                    weight_multiplier=float(rule_data.get("weight_multiplier", 1.0)),
                    time_window_seconds=int(rule_data.get("time_window_seconds", 5)),
                    minimum_confidence=float(rule_data.get("minimum_confidence", 0.5))
                )
            except (ValueError, TypeError, KeyError) as e:
                self._logger.warning(f"Invalid combination rule {rule_name}: {e}")
        
        return SystemConfiguration(
            detection_thresholds=detection_thresholds,
            correlation_window_seconds=int(config_data.get("correlation_window_seconds", 5)),
            max_correlation_events=int(config_data.get("max_correlation_events", 10)),
            alert_combination_rules=combination_rules,
            evidence_retention_days=int(config_data.get("evidence_retention_days", 30)),
            max_evidence_file_size_mb=int(config_data.get("max_evidence_file_size_mb", 100)),
            deployment_mode=deployment_mode,
            max_processing_latency_ms=int(config_data.get("max_processing_latency_ms", 500)),
            enable_gpu_acceleration=bool(config_data.get("enable_gpu_acceleration", True))
        )
    
    def _check_for_updates(self) -> None:
        """Check if configuration file has been updated and reload if necessary."""
        if not self._config_file_path or not self._config_file_path.exists():
            return
        
        try:
            current_mtime = self._config_file_path.stat().st_mtime
            if self._last_modified and current_mtime > self._last_modified:
                self._logger.info("Configuration file updated, reloading...")
                self._load_configuration()
        except OSError:
            # File might have been deleted or become inaccessible
            pass
            pass