"""
Phase 1: Compress and label all generated pages
"""
import os
import lz4.frame
import pandas as pd
from tqdm import tqdm

PAGE_SIZE = 4096
OUTPUT_DIR = "pages"
LABEL_CSV = "page_labels.csv"
ALPHA = 0.7  # Compressibility threshold

def label_pages():
    """Compress each page and assign ground truth label"""
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Error: '{OUTPUT_DIR}/' not found. Run phase1_generate_pages.py first.")
        return
    
    print(f"Compressing pages with LZ4-HC (compression_level=16)...")
    print(f"Compressibility threshold: α = {ALPHA}")
    
    records = []
    page_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.bin')])
    
    for filename in tqdm(page_files, desc="Processing pages"):
        path = os.path.join(OUTPUT_DIR, filename)
        
        with open(path, "rb") as f:
            page = f.read()
        
        # Verify page size
        if len(page) != PAGE_SIZE:
            print(f"⚠️  Warning: {filename} is {len(page)} bytes, expected {PAGE_SIZE}")
            continue
        
        # Compress with LZ4-HC
        compressed = lz4.frame.compress(page, compression_level=16)
        compressed_size = len(compressed)
        ratio = compressed_size / PAGE_SIZE
        
        # Assign label
        label = "compressible" if ratio <= ALPHA else "incompressible"
        
        # Extract page type from filename
        page_type = filename.split('_')[0]
        
        records.append({
            "page_id": filename,
            "page_type": page_type,
            "original_size": PAGE_SIZE,
            "compressed_size": compressed_size,
            "compression_ratio": ratio,
            "label": label
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(LABEL_CSV, index=False)
    
    print(f"\n✅ Labeled {len(df)} pages")
    print(f"📄 Saved to '{LABEL_CSV}'")
    
    # Summary statistics
    print("\n" + "="*50)
    print("LABEL DISTRIBUTION")
    print("="*50)
    print(df['label'].value_counts())
    print()
    
    print("="*50)
    print("COMPRESSION RATIO BY PAGE TYPE")
    print("="*50)
    summary = df.groupby('page_type').agg({
        'compression_ratio': ['mean', 'min', 'max', 'std']
    }).round(4)
    print(summary)
    print()
    
    print("="*50)
    print("COMPRESSIBILITY BY PAGE TYPE")
    print("="*50)
    comp_by_type = df.groupby(['page_type', 'label']).size().unstack(fill_value=0)
    print(comp_by_type)
    print()

if __name__ == "__main__":
    label_pages()
