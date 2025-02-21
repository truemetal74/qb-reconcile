import pandas as pd
from datetime import datetime
import argparse
import yaml
from typing import Dict, List, Tuple
import os

class TransactionMatcher:
    def __init__(self, config_path: str):
        # Load configuration from YAML file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _standardize_amount(self, row: pd.Series, file_config: Dict) -> float:
        """Calculate standardized transaction amount from charge or payment."""
        charge_col = file_config.get('charge_amount')
        payment_col = file_config.get('payment_amount')
        
        amount = 0.0
        if charge_col and pd.notna(row[charge_col]):
            # Remove commas and convert to float
            amount = float(str(row[charge_col]).replace(',', ''))
        elif payment_col and pd.notna(row[payment_col]):
            # Remove commas and convert to float
            amount = -float(str(row[payment_col]).replace(',', ''))
        return amount

    def _parse_date(self, date_str: str) -> datetime:
        """Try parsing date with multiple formats."""
        date_formats = ['%Y-%m-%d', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse date: {date_str}")

    def _load_and_standardize_file(self, file_path: str, file_config: Dict) -> pd.DataFrame:
        """Load CSV and standardize columns."""
        # Get number of rows to skip from config, default to 0
        skip_rows = file_config.get('skip_rows', 0)
        
        # Load CSV with specified number of rows to skip
        df = pd.read_csv(file_path, skiprows=skip_rows)
        
        # Create standardized columns
        standardized = pd.DataFrame()
        standardized['description'] = df[file_config['description']]
        standardized['date'] = df[file_config['date']].apply(self._parse_date)
        standardized['amount'] = df.apply(lambda row: self._standardize_amount(row, file_config), axis=1)
        standardized['source_data'] = df.apply(lambda row: dict(row), axis=1)
        
        return standardized

    def find_mismatches(self, file1_path: str, file2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare transactions and return mismatches."""
        # Get base filenames without path
        file1_name = os.path.basename(file1_path)
        file2_name = os.path.basename(file2_path)
        
        # Load and standardize both files
        df1 = self._load_and_standardize_file(file1_path, self.config['file1'])
        df2 = self._load_and_standardize_file(file2_path, self.config['file2'])

        # Find transactions unique to each file and duplicates
        # Only merge on amount, drop other columns before merging
        df1_merge = df1[['amount']]
        df2_merge = df2[['amount']]
        merged = df1_merge.merge(df2_merge, on=['amount'], how='outer', indicator=True)
        
        # Count occurrences in each file
        df1_counts = df1.groupby(['amount']).size().reset_index(name='count1')
        df2_counts = df2.groupby(['amount']).size().reset_index(name='count2')
        
        # Get descriptions and dates for each amount
        def get_desc_dates(df):
            from collections import Counter
            # Create tuples of (date, description) and count them
            date_desc_pairs = list(zip(df['date'], df['description']))
            counts = Counter(date_desc_pairs)
            return [(date, desc, count) for (date, desc), count in counts.items()]
            
        # Group by amount and collect date/description info
        df1_info = df1.groupby('amount').agg({
            'date': list,
            'description': list
        }).reset_index()
        df1_info['info1'] = df1_info.apply(lambda x: get_desc_dates(pd.DataFrame({
            'date': x['date'],
            'description': x['description']
        })), axis=1)
        df1_info = df1_info[['amount', 'info1']]

        df2_info = df2.groupby('amount').agg({
            'date': list,
            'description': list
        }).reset_index()
        df2_info['info2'] = df2_info.apply(lambda x: get_desc_dates(pd.DataFrame({
            'date': x['date'],
            'description': x['description']
        })), axis=1)
        df2_info = df2_info[['amount', 'info2']]
        
        # Merge all information back
        merged = merged.merge(df1_counts, on=['amount'], how='left')
        merged = merged.merge(df2_counts, on=['amount'], how='left')
        merged = merged.merge(df1_info, on=['amount'], how='left')
        merged = merged.merge(df2_info, on=['amount'], how='left')
        
        # Find mismatches
        only_in_file1 = merged[merged['_merge'] == 'left_only']
        only_in_file2 = merged[merged['_merge'] == 'right_only']
        
        # Find duplicates (where counts don't match)
        duplicates = merged[
            (merged['_merge'] == 'both') & 
            (merged['count1'] != merged['count2'])
        ]
        
        # Remove duplicate rows based on amount
        duplicates = duplicates.drop_duplicates(subset=['amount'])
        
        if not duplicates.empty:
            print("\nDuplicate transactions (count mismatch):")
            for _, row in duplicates.iterrows():
                print(f"\nAmount: {row['amount']}")
                print(f"Count in {file1_name}: {row['count1']}, Count in {file2_name}: {row['count2']}")
                print(f"\n{file1_name} entries:")
                for date, desc, count in row['info1']:
                    if count > 1:
                        print(f"  {date.strftime('%Y-%m-%d')}: {desc} (x{count})")
                    else:
                        print(f"  {date.strftime('%Y-%m-%d')}: {desc}")
                print(f"\n{file2_name} entries:")
                for date, desc, count in row['info2']:
                    if count > 1:
                        print(f"  {date.strftime('%Y-%m-%d')}: {desc} (x{count})")
                    else:
                        print(f"  {date.strftime('%Y-%m-%d')}: {desc}")

        return only_in_file1, only_in_file2

def match_transactions(bank_transactions, qb_transactions):
    matches = []
    mismatches = []
    
    # Create a dictionary to track how many times each bank transaction has been matched
    bank_match_counts = {i: 0 for i in range(len(bank_transactions))}
    
    # For each QB transaction, try to find a matching bank transaction
    for qb_idx, qb_trans in enumerate(qb_transactions):
        found_match = False
        
        # Extract QB transaction details
        qb_date = datetime.strptime(qb_trans[0], '%m/%d/%Y')
        qb_amount = float(qb_trans[4] or 0) - float(qb_trans[5] or 0)  # Debit - Credit
        qb_desc = qb_trans[2] or ''
        
        # Look for matching bank transaction
        for bank_idx, bank_trans in enumerate(bank_transactions):
            # Extract bank transaction details
            bank_date = datetime.strptime(bank_trans[0], '%Y-%m-%d')
            bank_amount = float(bank_trans[5] or 0) - float(bank_trans[6] or 0)  # Debit - Credit
            bank_desc = bank_trans[3]
            
            # Check if transactions match
            if (abs((qb_date - bank_date).days) <= 3 and
                abs(qb_amount - bank_amount) < 0.01 and
                (bank_desc in qb_desc or qb_desc in bank_desc)):
                
                # If this bank transaction was already matched, flag as mismatch
                if bank_match_counts[bank_idx] > 0:
                    mismatches.append({
                        'qb_trans': qb_trans,
                        'bank_trans': bank_trans,
                        'reason': 'Duplicate QB transaction'
                    })
                else:
                    matches.append({
                        'qb_trans': qb_trans,
                        'bank_trans': bank_trans
                    })
                    bank_match_counts[bank_idx] += 1
                
                found_match = True
                break
        
        if not found_match:
            mismatches.append({
                'qb_trans': qb_trans,
                'bank_trans': None,
                'reason': 'No matching bank transaction found'
            })
    
    # Check for unmatched bank transactions
    for bank_idx, bank_trans in enumerate(bank_transactions):
        if bank_match_counts[bank_idx] == 0:
            mismatches.append({
                'qb_trans': None,
                'bank_trans': bank_trans,
                'reason': 'No matching QB transaction found'
            })
    
    return matches, mismatches

def main():
    parser = argparse.ArgumentParser(description='Compare transactions from two CSV files')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML configuration file (default: config.yaml)')
    parser.add_argument('file1', help='Path to first CSV file')
    parser.add_argument('file2', help='Path to second CSV file')
    args = parser.parse_args()

    matcher = TransactionMatcher(args.config)
    missing_file1, missing_file2 = matcher.find_mismatches(args.file1, args.file2)

    # Debug: print column names
    print("\nColumns in missing_file1:", missing_file1.columns.tolist())
    print("\nColumns in missing_file2:", missing_file2.columns.tolist())

    print("\nTransactions only in file 1:")
    print(missing_file1[['amount', 'count1', 'info1']].to_string())
    
    print("\nTransactions only in file 2:")
    print(missing_file2[['amount', 'count2', 'info2']].to_string())

if __name__ == "__main__":
    main()