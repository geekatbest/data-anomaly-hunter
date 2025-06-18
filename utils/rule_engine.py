import yaml

class RuleEngine:
    def __init__(self, rule_file='rules/rules.yaml'):
        with open(rule_file, 'r') as f:
            self.rules = yaml.safe_load(f)['rules']

    def set_rules(self, new_rules):
        self.rules = new_rules

    def apply_rules(self, df):
        df['rule_label'] = None
        for rule in self.rules:
            try:
                condition = rule['if']
                label = rule['label']
                df.loc[df.eval(condition), 'rule_label'] = label
            except Exception as e:
                print(f"Rule error: {e}")
        df['rule_anomaly'] = df['rule_label'].notna().astype(int)
        return df
