"""
Phase 1: Generate diverse memory pages for compression analysis
"""
import os
import numpy as np

PAGE_SIZE = 4096
NUM_PAGES_PER_TYPE = 1000
OUTPUT_DIR = "pages"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_pages():
    """Generate diverse page types"""
    
    print(f"Generating {NUM_PAGES_PER_TYPE} pages per type...")
    
    # Type 1: Zero pages (highly compressible)
    print("  - Zero pages...")
    for i in range(NUM_PAGES_PER_TYPE):
        data = np.zeros(PAGE_SIZE, dtype=np.uint8).tobytes()
        with open(f"{OUTPUT_DIR}/zero_{i:04d}.bin", "wb") as f:
            f.write(data)
    
    # Type 2: Repeated pattern (highly compressible)
    print("  - Repeated pattern pages...")
    for i in range(NUM_PAGES_PER_TYPE):
        pattern = np.array([0xAB] * PAGE_SIZE, dtype=np.uint8).tobytes()
        with open(f"{OUTPUT_DIR}/repeat_{i:04d}.bin", "wb") as f:
            f.write(pattern)
    
    # Type 3: Random data (incompressible)
    print("  - Random pages...")
    for i in range(NUM_PAGES_PER_TYPE):
        rand_data = np.random.randint(0, 256, PAGE_SIZE, dtype=np.uint8).tobytes()
        with open(f"{OUTPUT_DIR}/rand_{i:04d}.bin", "wb") as f:
            f.write(rand_data)
    
    # Type 4: Structured data (mixed compressibility)
    print("  - Structured pages...")
    for i in range(NUM_PAGES_PER_TYPE):
        # Simulate realistic memory: some structure + some randomness
        structured = np.zeros(PAGE_SIZE, dtype=np.uint8)
        
        # Add some repeated patterns (like pointers, headers)
        structured[0:512] = 0x00  # header-like zeros
        structured[512:1024] = np.random.choice([0x01, 0x02, 0xFF], 512)  # limited values
        structured[1024:] = np.random.randint(0, 256, PAGE_SIZE - 1024)  # random data
        
        with open(f"{OUTPUT_DIR}/struct_{i:04d}.bin", "wb") as f:
            f.write(structured.tobytes())
    
    # Type 5: Sparse data (moderately compressible)
    print("  - Sparse pages...")
    for i in range(NUM_PAGES_PER_TYPE):
        sparse = np.zeros(PAGE_SIZE, dtype=np.uint8)
        # Only 10% non-zero
        num_nonzero = PAGE_SIZE // 10
        indices = np.random.choice(PAGE_SIZE, num_nonzero, replace=False)
        sparse[indices] = np.random.randint(1, 256, num_nonzero)
        
        with open(f"{OUTPUT_DIR}/sparse_{i:04d}.bin", "wb") as f:
            f.write(sparse.tobytes())
    
    total_pages = NUM_PAGES_PER_TYPE * 5
    print(f"\n✅ Generated {total_pages} pages in '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    generate_pages()
