"""
Phase 1B: Collect Real-World Page Data
Extract 4KB pages from real files (text, images, binaries)
"""
import os
import lz4.frame
import pandas as pd
import numpy as np
from tqdm import tqdm

PAGE_SIZE = 4096
ALPHA = 0.7
OUTPUT_DIR = "pages_real"
MAX_PAGES_PER_SOURCE = 500  # Limit pages per source type

os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_pages_from_file(filepath, source_type, page_counter, records):
    """Extract 4KB pages from a file"""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        
        # Split into 4KB pages
        pages_extracted = 0
        for i in range(0, len(data), PAGE_SIZE):
            if pages_extracted >= MAX_PAGES_PER_SOURCE:
                break
                
            page = data[i:i+PAGE_SIZE]
            if len(page) < PAGE_SIZE:
                # Pad partial page with zeros
                page = page + b'\x00' * (PAGE_SIZE - len(page))
            
            # Compress and label
            compressed = lz4.frame.compress(page, compression_level=16)
            ratio = len(compressed) / PAGE_SIZE
            label = "compressible" if ratio <= ALPHA else "incompressible"
            
            # Save page
            page_id = f"{source_type}_{page_counter:04d}.bin"
            page_path = os.path.join(OUTPUT_DIR, page_id)
            with open(page_path, "wb") as fpage:
                fpage.write(page)
            
            records.append({
                "page_id": page_id,
                "source_type": source_type,
                "source_file": os.path.basename(filepath),
                "original_size": PAGE_SIZE,
                "compressed_size": len(compressed),
                "compression_ratio": ratio,
                "label": label
            })
            
            page_counter += 1
            pages_extracted += 1
        
        return page_counter
    
    except Exception as e:
        print(f"  ⚠️  Error reading {filepath}: {e}")
        return page_counter

def collect_from_python_files():
    """Collect pages from Python source files (text)"""
    print("Collecting from Python source files...")
    records = []
    page_counter = 0
    
    # Collect from our own project files
    for root, dirs, files in os.walk("."):
        # Skip virtual environment and cache directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'pages', 'pages_real']]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                page_counter = collect_pages_from_file(filepath, "text", page_counter, records)
                
                if page_counter >= MAX_PAGES_PER_SOURCE:
                    break
        
        if page_counter >= MAX_PAGES_PER_SOURCE:
            break
    
    print(f"  Collected {len(records)} text pages")
    return records

def collect_from_csv_files():
    """Collect pages from CSV files (structured text)"""
    print("Collecting from CSV files...")
    records = []
    page_counter = 0
    
    csv_files = [f for f in os.listdir(".") if f.endswith('.csv')]
    
    for csv_file in csv_files:
        page_counter = collect_pages_from_file(csv_file, "csv", page_counter, records)
        
        if page_counter >= MAX_PAGES_PER_SOURCE:
            break
    
    print(f"  Collected {len(records)} CSV pages")
    return records

def collect_from_binary_files():
    """Collect pages from binary files"""
    print("Collecting from binary page files...")
    records = []
    page_counter = 0
    
    # Use existing synthetic pages as "binary" data
    if os.path.exists("pages"):
        bin_files = [f for f in os.listdir("pages") if f.endswith('.bin')]
        
        # Sample random binary files
        sampled_files = np.random.choice(bin_files, min(MAX_PAGES_PER_SOURCE, len(bin_files)), replace=False)
        
        for bin_file in sampled_files:
            filepath = os.path.join("pages", bin_file)
            page_counter = collect_pages_from_file(filepath, "binary", page_counter, records)
    
    print(f"  Collected {len(records)} binary pages")
    return records

def collect_from_markdown_files():
    """Collect pages from Markdown documentation"""
    print("Collecting from Markdown files...")
    records = []
    page_counter = 0
    
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'pages', 'pages_real']]
        
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                page_counter = collect_pages_from_file(filepath, "markdown", page_counter, records)
                
                if page_counter >= MAX_PAGES_PER_SOURCE:
                    break
        
        if page_counter >= MAX_PAGES_PER_SOURCE:
            break
    
    print(f"  Collected {len(records)} markdown pages")
    return records

def main():
    print("="*70)
    print("PHASE 1B: REAL-WORLD PAGE DATA COLLECTION")
    print("="*70)
    print()
    print(f"Target: {MAX_PAGES_PER_SOURCE} pages per source type")
    print(f"Page size: {PAGE_SIZE} bytes")
    print(f"Compressibility threshold: α = {ALPHA}")
    print()
    
    all_records = []
    
    # Collect from different sources
    all_records.extend(collect_from_python_files())
    all_records.extend(collect_from_csv_files())
    all_records.extend(collect_from_markdown_files())
    all_records.extend(collect_from_binary_files())
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    if len(df) == 0:
        print("\n❌ No pages collected. Please ensure source files exist.")
        return
    
    # Save labels
    df.to_csv("page_labels_real.csv", index=False)
    
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"Total pages collected: {len(df)}")
    print()
    
    print("By source type:")
    print(df['source_type'].value_counts())
    print()
    
    print("By label:")
    print(df['label'].value_counts())
    print()
    
    print("Compression ratio statistics:")
    print(df['compression_ratio'].describe())
    print()
    
    print("By source type and label:")
    print(pd.crosstab(df['source_type'], df['label']))
    print()
    
    print(f"✅ Saved {len(df)} pages to '{OUTPUT_DIR}/'")
    print(f"✅ Saved labels to 'page_labels_real.csv'")
    
    # Sanity checks
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    for source_type in df['source_type'].unique():
        subset = df[df['source_type'] == source_type]
        avg_ratio = subset['compression_ratio'].mean()
        print(f"{source_type:10s}: avg ratio = {avg_ratio:.4f}")
    
    print()
    print("✅ Real-world dataset ready for evaluation!")

if __name__ == "__main__":
    main()
