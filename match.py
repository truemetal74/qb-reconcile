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
        # Get skip rows value outside try block so it's available in except
        skiprows = file_config.get('skip_rows', 0)
        # Determine if this is a bank or QB file based on the config
        file_type = "QB" if file_config.get('description') == "Memo" else "bank"
        
        try:
            # Load the file with proper skip_rows
            df = pd.read_csv(file_path, skiprows=skiprows)
            
            # Rest of the function...
            date_col = file_config.get('date', 'Date')
            if date_col not in df.columns:
                raise KeyError(f"Date column '{date_col}' not found in {file_type} CSV.\nColumns in file (after skipping {skiprows} rows): {', '.join(df.columns)}")
            df['date'] = df[date_col].apply(self._standardize_date)
            
            desc_col = file_config.get('description', 'Description')
            if desc_col not in df.columns:
                raise KeyError(f"Description column '{desc_col}' not found in {file_type} CSV.\nColumns in file (after skipping {skiprows} rows): {', '.join(df.columns)}")
            df['description'] = df[desc_col].fillna('')
            
            amount_col = file_config.get('amount_column')
            charge_col = file_config.get('charge_amount')
            payment_col = file_config.get('payment_amount')
            
            if amount_col and amount_col not in df.columns:
                raise KeyError(f"Amount column '{amount_col}' not found in {file_type} CSV.\nColumns in file (after skipping {skiprows} rows): {', '.join(df.columns)}")
            if charge_col and charge_col not in df.columns:
                raise KeyError(f"Charge column '{charge_col}' not found in {file_type} CSV.\nColumns in file (after skipping {skiprows} rows): {', '.join(df.columns)}")
            if payment_col and payment_col not in df.columns:
                raise KeyError(f"Payment column '{payment_col}' not found in {file_type} CSV.\nColumns in file (after skipping {skiprows} rows): {', '.join(df.columns)}")
            
            df['amount'] = df.apply(lambda row: self._standardize_amount(row, file_config), axis=1)
            
            return df
            
        except Exception as e:
            print(f"\nError processing {file_type} file: {file_path}")
            print(f"Please check your config.yaml file and ensure column names match the CSV headers.")
            
            # Always show the actual headers after skipping rows
            try:
                df_actual = pd.read_csv(file_path, skiprows=skiprows)
                print(f"\nActual columns in {file_type} file (after skipping {skiprows} rows):")
                print(', '.join(df_actual.columns))
            except Exception as read_error:
                print(f"Error reading {file_type} file: {str(read_error)}")
            
            raise

    def find_mismatches(self, file1_path: str, file2_path: str) -> pd.DataFrame:
        """Compare transactions and return mismatches."""
        # Load and standardize both files
        df1 = self._load_and_standardize_file(file1_path, self.config['bank'])  # bank file with bank config
        df2 = self._load_and_standardize_file(file2_path, self.config['qb'])    # qb file with qb config

        # Round amounts to cents
        df1['amount'] = df1['amount'].round(2)
        df2['amount'] = df2['amount'].round(2)

        # First group by amount to find potential matches
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
            [(d, desc, 'bank') for d, desc in zip(row['date1'], row['description1'])]
            if isinstance(row['date1'], list) else [], axis=1)
        merged['info2'] = merged.apply(lambda row: 
            [(d, desc, 'qb') for d, desc in zip(row['date2'], row['description2'])]
            if isinstance(row['date2'], list) else [], axis=1)

        # Find mismatches (where counts don't match)
        mismatches = merged[merged['count1'] != merged['count2']][['amount', 'count1', 'count2', 'info1', 'info2']]

        return mismatches

def main():
    parser = argparse.ArgumentParser(description='Compare transactions from bank statement and QuickBooks files')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML configuration file (default: config.yaml)')
    parser.add_argument('bank_file', help='Path to bank statement CSV file')
    parser.add_argument('qb_file', help='Path to QuickBooks CSV file')
    args = parser.parse_args()

    matcher = TransactionMatcher(args.config)
    mismatches = matcher.find_mismatches(args.bank_file, args.qb_file)

    print("\nMismatched transaction counts:")
    print("     Amount    QB  Bank")
    print("     ------  ---- ----")
    for idx, row in mismatches.iterrows():
        amount = f"{row['amount']:8.2f}"
        qb_count = int(row['count2'])
        bank_count = int(row['count1'])
        
        print(f"{idx:3d} {amount} {qb_count:4d} {bank_count:4d}")
        
        # Print bank transactions (if any)
        if bank_count > 0:
            print("     Bank transactions:")
            for date1, desc1, _ in row['info1']:
                print(f"     {date1.strftime('%Y-%m-%d')}: {desc1}")
        
        # Print QB transactions (if any)
        if qb_count > 0:
            print("     QB transactions:")
            for date2, desc2, _ in row['info2']:
                print(f"     {date2.strftime('%Y-%m-%d')}: {desc2}")
        
        print()  # Add blank line between different amounts

if __name__ == "__main__":
    main()