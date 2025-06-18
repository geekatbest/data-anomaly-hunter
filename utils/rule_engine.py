import pandas as pd
import yaml

class RuleEngine:
    def __init__(self, rule_file='rules/rules.yaml'):
        with open(rule_file, 'r') as f:
            self.rules = yaml.safe_load(f)['rules']

    def apply_rules(self, df):
        labels = []
        for rule in self.rules:
            condition = rule['if']
            label = rule['label']
            match = df.eval(condition)
            df.loc[match, 'rule_label'] = label
            labels.append(label)
        df['rule_anomaly'] = df['rule_label'].notna().astype(int)
        return df
