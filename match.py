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
        # Check if using single amount column or separate charge/payment columns
        amount_col = file_config.get('amount_column')
        if amount_col:
            # Single amount column
            amount = float(str(row[amount_col]).replace(',', ''))
            charges_are_negative = file_config.get('charges_are_negative', True)
            
            if charges_are_negative:
                # If amount is negative, it's a charge - make it positive
                # If amount is positive, it's a payment/credit - make it negative
                return -amount
            else:
                # If amount is negative, it's a payment/credit - keep it negative
                # If amount is positive, it's a charge - keep it positive
                return amount
        else:
            # Separate charge and payment columns
            charge_col = file_config.get('charge_amount')
            payment_col = file_config.get('payment_amount')
            
            amount = 0.0
            if charge_col and pd.notna(row[charge_col]):
                amount = float(str(row[charge_col]).replace(',', ''))
            elif payment_col and pd.notna(row[payment_col]):
                amount = -float(str(row[payment_col]).replace(',', ''))
            return amount

    def _standardize_date(self, date_str: str) -> datetime:
        """Convert date string to datetime object."""
        try:
            # Try MM/DD/YYYY format first (for Chase bank statements)
            return pd.to_datetime(date_str, format='%m/%d/%Y')
        except ValueError:
            try:
                # Try YYYY-MM-DD format (for QuickBooks)
                return pd.to_datetime(date_str)
            except ValueError:
                raise ValueError(f"Unable to parse date: {date_str}")

    def _load_and_standardize_file(self, file_path: str, file_config: Dict) -> pd.DataFrame:
        """Load CSV file and standardize the data format."""
        # Skip rows if specified
        skiprows = file_config.get('skip_rows', 0)
        df = pd.read_csv(file_path, skiprows=skiprows)
        
        # Standardize date
        date_col = file_config.get('date', 'Date')
        df['date'] = df[date_col].apply(self._standardize_date)
        
        # Standardize description
        desc_col = file_config.get('description', 'Description')
        df['description'] = df[desc_col].fillna('')
        
        # Standardize amount
        df['amount'] = df.apply(lambda row: self._standardize_amount(row, file_config), axis=1)
        
        return df

    def find_mismatches(self, file1_path: str, file2_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare transactions and return mismatches."""
        # Load and standardize both files
        df1 = self._load_and_standardize_file(file1_path, self.config['file1'])
        df2 = self._load_and_standardize_file(file2_path, self.config['file2'])

        # Round amounts to cents
        df1['amount'] = df1['amount'].round(2)
        df2['amount'] = df2['amount'].round(2)

        # Group by amount and count occurrences in each file
        counts1 = df1.groupby('amount').agg({
            'date': list,
            'description': list,
        }).reset_index()
        counts1['count1'] = counts1['description'].str.len()
        
        counts2 = df2.groupby('amount').agg({
            'date': list,
            'description': list,
        }).reset_index()
        counts2['count2'] = counts2['description'].str.len()

        # Merge on amount
        merged = pd.merge(counts1, counts2, on=['amount'], how='outer', suffixes=('1', '2'))
        
        # Fill NaN counts with 0
        merged['count1'] = merged['count1'].fillna(0)
        merged['count2'] = merged['count2'].fillna(0)
        
        # Create info columns with dates and descriptions
        merged['info1'] = merged.apply(lambda row: 
            [(d, desc, 1) for d, desc in zip(row['date1'], row['description1'])]
            if isinstance(row['description1'], list) else [], axis=1)
        merged['info2'] = merged.apply(lambda row: 
            [(d, desc, 1) for d, desc in zip(row['date2'], row['description2'])]
            if isinstance(row['description2'], list) else [], axis=1)

        # Find mismatches (where count1 != count2)
        mismatches1 = merged[merged['count1'] > merged['count2']][['amount', 'count1', 'count2', 'info1']]
        mismatches2 = merged[merged['count2'] > merged['count1']][['amount', 'count2', 'count1', 'info2']]

        return mismatches1, mismatches2

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
    parser = argparse.ArgumentParser(description='Compare transactions from bank statement and QuickBooks files')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML configuration file (default: config.yaml)')
    parser.add_argument('bank_file', help='Path to bank statement CSV file')
    parser.add_argument('qb_file', help='Path to QuickBooks CSV file')
    args = parser.parse_args()

    matcher = TransactionMatcher(args.config)
    missing_file1, missing_file2 = matcher.find_mismatches(args.bank_file, args.qb_file)

    def format_info(info_list):
        return '; '.join(f"{date.strftime('%Y-%m-%d')}: {desc}" for date, desc, _ in info_list)

    def print_mismatches(df, title, found_col, expected_col):
        print(f"\n{title}")
        print("     Amount    QB  Bank  Info")
        print("     ------  ---- ----  ----")
        for idx, row in df.iterrows():
            amount = f"{row['amount']:8.2f}"
            qb_count = int(row['count2'])
            bank_count = int(row['count1'])
            info = row['info']
            print(f"{idx:3d} {amount} {qb_count:4d} {bank_count:4d}  {info}")

    # Prepare and print bank file mismatches
    missing_file1['info'] = missing_file1['info1'].apply(format_info)
    print_mismatches(missing_file1, "Transactions only in bank file", 'count1', 'count2')
    
    # Prepare and print QuickBooks file mismatches
    missing_file2['info'] = missing_file2['info2'].apply(format_info)
    print_mismatches(missing_file2, "Transactions only in QuickBooks file", 'count2', 'count1')

if __name__ == "__main__":
    main()