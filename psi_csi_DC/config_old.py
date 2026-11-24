class Config:
    PROD_PROJ_ID = 'prj-prod-dataplatform'
    TEST_PROJ_ID = 'prj-tonikbank-u-datalake-34e9'
    MODEL_CONFIG = {
        'sil_model',
        'cash_model'
        }
    SIL_MODEL_CONFIG = {
        
    "cic_sil_model": {
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
                        "app_cnt_finance_90d", #NA in Train
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
                        "app_cnt_finance_90d": "float", #NA in Train
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
                'where_condition': "('Beta-Cash-AppScore-Model', 'apps_score_cash')"},
            'v1.1': {
                'where_condition': "('Beta-Cash-AppScore-Model', 'apps_score_cash')"}
            },
        "alpha_stack_cash_model": {
            'v1': {
                'where_condition': "('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')"},
            'v1.1': {
                'where_condition': "('Alpha-Cash-Stack-Model', 'alpha_stack_model_cash')"}
        },
        "cic_cash_model": {
            'v1': {
                'where_condition': "('Alpha-Cash-CIC-Model','Alpha Cash CIC Model','cic_model_cash')"}},
        "beta_stack_cash_model": {
            'v1': {
                'where_condition': "('Beta-Cash-Stack-Model', 'beta_stack_model_cash')"},
            'v1.1': {
                'where_condition': "('Beta-Cash-Stack-Model', 'beta_stack_model_cash')"}
        },
        "beta_demo_cash_model": {
            'v1': {
                'where_condition': "('Beta-Cash-Demo-Model', 'beta_demo_model_cash')"},
            },
    }