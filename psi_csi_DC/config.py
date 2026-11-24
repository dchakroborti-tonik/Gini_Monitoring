"""
Merged configuration file — combines the "current" config (latest/production
values) with the previous config file that still had v1-labeled model
versions.

Merge approach:
- MODEL_CONFIG: taken from the current file (superset of active models).
- For every model in SIL_MODEL_CONFIG and CASH_MODEL_CONFIG, all version keys
  from BOTH source files are kept side by side (e.g. "v1" and "v2", or "v1"
  and "v1.1") so no history is lost.
- Where the SAME version key exists in both files (every "v2" SIL model, and
  the "v1"/"v1.1" cash models), the CURRENT file's definition is used as-is —
  it's treated as the authoritative/live config. The old file's extra fields
  for that same version (see point 1 below) are NOT auto-merged in.
- Items that only appeared in the old file (every SIL "v1" entry, and the
  cash "credo_score_model" entry) are carried over unchanged.

Things worth double-checking (flagged inline with NOTE comments too):
1. The old file had test/prod date-range fields on every version, including
   the "v2" SIL entries and all cash-model entries. The current file dropped
   these fields entirely. They are NOT re-added here for current-sourced
   versions since it's unclear if the removal was intentional — say the word
   if you want them restored.
2. beta_demo_sil_model["v2"]["calcfeature_config"]["prod"] uses the key
   "feature_types_json" instead of "feature_types" (every other train/prod
   block uses "feature_types"). Carried over as-is from your current file.
3. credo_score_model (cash) only existed in the old file and has a
   structurally different / likely broken shape vs. its siblings (a "v1" key
   containing its own nested "v1"/"v1.1" sub-keys, and a where_condition that
   duplicates beta_stack_model_cash rather than referencing credo). Carried
   over as-is — please verify before relying on it.
"""


class Config:
    PROD_PROJ_ID = 'prj-prod-dataplatform'
    TEST_PROJ_ID = 'prj-tonikbank-u-datalake-34e9'

    MODEL_CONFIG = {
        'sil_model',
        'cash_model'
    }

    SIL_MODEL_CONFIG = {
        "cic_sil_model": {
            # --- legacy version, only present in the old config file ---
            "v1": {
                "test_start_date": "2023-01-01",
                "test_end_date": "2025-03-31",
                "prod_start_date": "2025-03-01",
                "prod_end_date": "2025-12-31",
                "where_condition": "'Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model'",
                "calcfeature_config": {
                    "common": {
                        "base_path": "$",
                        "features": [
                            "cic_vel_contract_nongranted_cnt_12on24",
                            # "cic_max_age_all_contracts",
                            # "cic_vel_contract_nongranted_amt_6on12",
                            # "cic_amt_nongranted_contracts_3M",
                            # "cic_avg_no_of_instal_24M"
                        ],
                        "feature_types": {
                            "cic_vel_contract_nongranted_cnt_12on24": "float64"
                        }
                    }
                }
            },
            # --- current version ---
            "v2": {
                "where_condition": "'Alpha - CIC-SIL-Model', 'cic_model_sil', 'Sil-Alpha-CIC-SIL-Model'",
                "calcfeature_config": {
                    "common": {
                        "base_path": "$",
                        "features": [
                            "Personal_Loans_granted_contracts_amt_24M",
                            "ScoreRange",
                            "has_ever_been_overdue",
                            "ln_loan_level_user_type",
                            "cnt_nongranted_contracts_3M"
                        ],
                        "feature_types": {
                            "Personal_Loans_granted_contracts_amt_24M": "string",
                            "ScoreRange": "string",
                            "has_ever_been_overdue": "string",
                            "ln_loan_level_user_type": "string",
                            "cnt_nongranted_contracts_3M": "string"
                        }
                    }
                }
            }
        },

        "alpha_stack_sil_model": {
            "v1": {
                "test_start_date": "2023-01-01",
                "test_end_date": "2025-03-31",
                "prod_start_date": "2025-03-01",
                "prod_end_date": "2025-12-31",
                "where_condition": "'Alpha - StackingModel', 'alpha_stack_model_sil'",
                "calcfeature_config": {
                    "common": {
                        "base_path": "",
                        "features": [
                            "sb_demo_score",
                            "s_cic_score",
                            "s_credo_score",
                            "s_apps_score"
                        ],
                        "feature_types": {
                            "sb_demo_score": "float64",
                            "s_cic_score": "float64",
                            "s_credo_score": "float64",
                            "s_apps_score": "float64"
                        }
                    }
                }
            },
            "v2": {
                "where_condition": "'Alpha - StackingModel', 'alpha_stack_model_sil'",
                "calcfeature_config": {
                    "common": {
                        "base_path": "",
                        "features": [
                            "sb_demo_score",
                            "sa_cic_score",
                            "s_credo_score",
                            "apps_score"
                        ],
                        "feature_types": {
                            "sb_demo_score": "string",
                            "sa_cic_score": "string",
                            "s_credo_score": "string",
                            "apps_score": "string"
                        }
                    }
                }
            }
        },

        "beta_stack_sil_model": {
            "v1": {
                "test_start_date": "2023-01-01",
                "test_end_date": "2025-03-31",
                "prod_start_date": "2025-03-01",
                "prod_end_date": "2025-12-31",
                "where_condition": "'Beta - StackScoreModel', 'beta_stack_model_sil'",
                "calcfeature_config": {
                    "train": {
                        "base_path": "$",
                        "features": [
                            "apps_score",
                            "credo_gen_score",
                            "beta_demo_score"
                        ],
                        "feature_types": {
                            "apps_score": "float64",
                            "credo_gen_score": "float64",
                            "beta_demo_score": "float64"
                        },
                        "aliases": {
                            "apps_score": "s_apps_score",
                            "credo_gen_score": "s_credo_score",
                            "beta_demo_score": "sb_demo_score"
                        }
                    },
                    "prod": {
                        "base_path": "$",
                        "features": [
                            "s_apps_score",
                            "s_credo_score",
                            "sb_demo_score"
                        ],
                        "feature_types": {
                            "s_apps_score": "float64",
                            "s_credo_score": "float64",
                            "sb_demo_score": "float64"
                        }
                    }
                }
            },
            "v2": {
                "where_condition": "'Beta - StackScoreModel', 'beta_stack_model_sil'",
                "calcfeature_config": {
                    "train": {
                        "base_path": "$",
                        "features": [
                            "apps_score",
                            "s_credo_score",
                            "sb_demo_score"
                        ],
                        "feature_types": {
                            "apps_score": "numeric",
                            "s_credo_score": "numeric",
                            "sb_demo_score": "numeric"
                        }
                    },
                    "prod": {
                        "base_path": "$.stackScoreModel",
                        "features": [
                            "apps_score",
                            "s_credo_score",
                            "sb_demo_score"
                        ],
                        "feature_types": {
                            "s_apps_score": "numeric",
                            "s_credo_score": "numeric",
                            "sb_demo_score": "numeric"
                        }
                    }
                }
            }
        },

        "beta_demo_sil_model": {
            "v1": {
                "test_start_date": "2023-01-01",
                "test_end_date": "2025-03-31",
                "prod_start_date": "2025-03-01",
                "prod_end_date": "2025-12-31",
                "where_condition": "'Beta - DemoScoreModel', 'beta_demo_model_sil'",
                "calcfeature_config": {
                    "common": {
                        "base_path": "$",
                        "features": [
                            "beta_de_ln_vas_opted_flag",
                            "beta_de_ln_doc_type_rolled",
                            "beta_de_ln_marital_status",
                            "beta_de_ln_age_bin",
                            "beta_de_ln_ref2_type"
                        ],
                        "feature_types": {
                            "beta_de_ln_vas_opted_flag": "string",
                            "beta_de_ln_doc_type_rolled": "string",
                            "beta_de_ln_marital_status": "string",
                            "beta_de_ln_age_bin": "string",
                            "beta_de_ln_ref2_type": "string"
                        }
                    }
                }
            },
            "v2": {
                "where_condition": "'Beta - DemoScoreModel', 'beta_demo_model_sil'",
                "calcfeature_config": {
                    "train": {
                        "base_path": "$",
                        "features": [
                            "ln_vas_opted_flag",
                            "ln_doc_type_rolled",
                            "ln_industry_new_cat_bin",
                            "ln_marital_status",
                            "ln_age"
                        ],
                        "feature_types": {
                            "ln_vas_opted_flag": "int",
                            "ln_doc_type_rolled": "string",
                            "ln_industry_new_cat_bin": "string",
                            "ln_marital_status": "string",
                            "ln_age": "int"
                        }
                    },
                    "prod": {
                        "base_path": "$.demoScoreModel",
                        "features": [
                            "ln_vas_opted_flag",
                            "ln_doc_type_rolled",
                            "ln_industry_new_cat_bin",
                            "ln_marital_status",
                            "ln_age"
                        ],
                        # NOTE: source used "feature_types_json" here (not
                        # "feature_types" like every other block) — kept as-is,
                        # verify this isn't a typo.
                        "feature_types_json": {
                            "ln_vas_opted_flag": "int",
                            "ln_doc_type_rolled": "string",
                            "ln_industry_new_cat_bin": "string",
                            "ln_marital_status": "string",
                            "ln_age": "int"
                        }
                    }
                }
            }
        },

        "apps_score_sil_model": {
            "v1": {
                "test_start_date": "2023-01-01",
                "test_end_date": "2025-03-31",
                "prod_start_date": "2025-03-01",
                "prod_end_date": "2025-12-31",
                "where_condition": "'Beta - AppsScoreModel', 'apps_score_model_sil'",
                "calcfeature_config": {
                    "common": {
                        "base_path": "$",
                        "features": [
                            "app_first_competitors_install_to_apply_days",
                            "app_median_time_bw_installed_mins_30d",
                            "app_cnt_absence_tag_90d",
                            "app_cnt_finance_90d",
                            "app_cnt_rated_for_3plus_ever"
                        ],
                        "feature_types": {
                            "app_first_competitors_install_to_apply_days": "float64",
                            "app_median_time_bw_installed_mins_30d": "float64",
                            "app_cnt_absence_tag_90d": "float64",
                            "app_cnt_finance_90d": "float64",
                            "app_cnt_rated_for_3plus_ever": "float64"
                        }
                    }
                }
            },
            "v2": {
                "where_condition": "'Beta - AppsScoreModel', 'apps_score_model_sil'",
                "calcfeature_config": {
                    "train": {
                        "base_path": "$",
                        "features": [
                            "app_cnt_payday_ever",
                            "app_cnt_finance_ever",
                            "app_cnt_competitors_sil_ever",
                            "app_cnt_competitors_ever",
                            "app_cnt_finance_365d",
                            "app_cnt_rated_for_3plus_ever",
                            "app_cnt_finance_90d",
                            "app_cnt_absence_tag_180d",
                            "app_cnt_productivity_ever",
                            "app_median_time_bw_installed_mins_3d"
                        ],
                        "feature_types": {
                            "app_cnt_payday_ever": "float",
                            "app_cnt_finance_ever": "float",
                            "app_cnt_competitors_sil_ever": "float",
                            "app_cnt_competitors_ever": "float",
                            "app_cnt_finance_365d": "float",
                            "app_cnt_rated_for_3plus_ever": "float",
                            "app_cnt_finance_90d": "float",
                            "app_cnt_absence_tag_180d": "float",
                            "app_cnt_productivity_ever": "float",
                            "app_median_time_bw_installed_mins_3d": "float"
                        }
                    },
                    "prod": {
                        "base_path": "$.appScoreModel",
                        "features": [
                            "app_cnt_payday_ever",
                            "app_cnt_finance_ever",
                            "app_cnt_competitors_sil_ever",
                            "app_cnt_competitors_ever",
                            "app_cnt_finance_365d",
                            "app_cnt_rated_for_3plus_ever",
                            "app_cnt_finance_90d",  # NA in Train
                            "app_cnt_absence_tag_180d",
                            "app_cnt_productivity_ever",
                            "app_median_time_bw_installed_mins_3d"
                        ],
                        "feature_types": {
                            "app_cnt_payday_ever": "float",
                            "app_cnt_finance_ever": "float",
                            "app_cnt_competitors_sil_ever": "float",
                            "app_cnt_competitors_ever": "float",
                            "app_cnt_finance_365d": "float",
                            "app_cnt_rated_for_3plus_ever": "float",
                            "app_cnt_finance_90d": "float",  # NA in Train
                            "app_cnt_absence_tag_180d": "float",
                            "app_cnt_productivity_ever": "float",
                            "app_median_time_bw_installed_mins_3d": "float"
                        }
                    }
                }
            }
        }
    }

    CASH_MODEL_CONFIG = {
        "apps_score_cash_model": {
            'v1': {
                'where_condition': "('Beta-Cash-AppScore-Model', 'apps_score_cash')"
            },
            'v1.1': {
                'where_condition': "('Beta-Cash-AppScore-Model', 'apps_score_cash')"
            }
        },
        "alpha_stack_cash_model": {
            'v1': {
                'where_condition': "('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')"
            },
            'v1.1': {
                'where_condition': "('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')"
            }
        },
        "cic_cash_model": {
            'v1': {
                'where_condition': "('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')"
            }
        },
        "beta_stack_cash_model": {
            'v1': {
                'where_condition': "('Beta-Cash-Stack-Model', 'beta_stack_model_cash')"
            },
            'v1.1': {
                'where_condition': "('Beta-Cash-Stack-Model', 'beta_stack_model_cash')"
            }
        },
        "beta_demo_cash_model": {
            'v1': {
                'where_condition': "('Beta-Cash-Demo-Model', 'beta_demo_model_cash')"
            }
        },
        # NOTE: this model only existed in the old config file, and its shape
        # doesn't match its siblings above — "v1" contains its own nested
        # "v1"/"v1.1" keys instead of dates living directly under "v1", and the
        # where_condition duplicates beta_stack_model_cash rather than
        # referencing credo. Carried over unchanged — please confirm this is
        # correct / still needed.
        "credo_score_model": {
            'v1': {
                'v1': {
                    'test_start_date': '2024-10-01',
                    'test_end_date': '2025-09-30'
                },
                'v1.1': {
                    'prod_start_date': '2025-12-01',
                    'prod_end_date': '2026-02-28'
                },
                'where_condition': "('Beta-Cash-Stack-Model', 'beta_stack_model_cash')"
            }
        }
    }