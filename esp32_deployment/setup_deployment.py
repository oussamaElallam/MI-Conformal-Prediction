"""
Setup script to prepare ESP32 deployment package.
Generates model files and copies them to the deployment folder.
"""
import os
import sys
import shutil

print("="*60)
print("ESP32-S3 Deployment Setup")
print("="*60)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)
EDGE_DIR = os.path.join(PARENT_DIR, 'edge')
MODEL_DIR = os.path.join(EDGE_DIR, 'model')
INCLUDE_DIR = os.path.join(PROJECT_ROOT, 'include')

os.makedirs(INCLUDE_DIR, exist_ok=True)

print(f"\nProject root: {PROJECT_ROOT}")
print(f"Edge directory: {EDGE_DIR}")
print(f"Model directory: {MODEL_DIR}")

# Step 1: Check if model files exist
print("\n1. Checking for model files...")
required_files = [
    'model_int8.tflite',
    'model_data.h',
    'cp_params.h'
]

missing_files = []
for file in required_files:
    filepath = os.path.join(MODEL_DIR, file)
    if os.path.exists(filepath):
        print(f"   ✓ Found: {file}")
    else:
        print(f"   ✗ Missing: {file}")
        missing_files.append(file)

# Step 2: Generate missing files
if missing_files:
    print("\n2. Generating missing model files...")
    print("   This will take a few minutes...\n")
    
    # Change to parent directory to run scripts
    os.chdir(PARENT_DIR)
    
    if 'model_int8.tflite' in missing_files or 'model_data.h' in missing_files:
        print("   → Running export_tflite_micro_model.py...")
        ret = os.system('python edge/export_tflite_micro_model.py')
        if ret != 0:
            print("   ERROR: Failed to export TFLite model")
            sys.exit(1)
        
        print("   → Running tflite_to_cc.py...")
        ret = os.system('python edge/tflite_to_cc.py')
        if ret != 0:
            print("   ERROR: Failed to convert model to C array")
            sys.exit(1)
    
    if 'cp_params.h' in missing_files:
        print("   → Running compute_cp_thresholds.py...")
        ret = os.system('python edge/compute_cp_thresholds.py')
        if ret != 0:
            print("   ERROR: Failed to compute conformal thresholds")
            sys.exit(1)
    
    print("   ✓ Model files generated successfully")
else:
    print("   ✓ All model files already exist")

# Step 3: Copy files to include directory
print("\n3. Copying files to deployment package...")

files_to_copy = [
    ('model_data.h', 'Model header'),
    ('cp_params.h', 'Conformal prediction parameters')
]

for filename, description in files_to_copy:
    src = os.path.join(MODEL_DIR, filename)
    dst = os.path.join(INCLUDE_DIR, filename)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"   ✓ Copied: {filename} ({description})")
    else:
        print(f"   ✗ ERROR: {filename} not found at {src}")
        sys.exit(1)

# Step 4: Verify deployment package
print("\n4. Verifying deployment package...")

required_deployment_files = [
    ('platformio.ini', 'PlatformIO configuration'),
    ('src/main.cpp', 'Main application code'),
    ('include/model_data.h', 'Model header'),
    ('include/cp_params.h', 'CP parameters'),
    ('README.md', 'Documentation')
]

all_ok = True
for filepath, description in required_deployment_files:
    full_path = os.path.join(PROJECT_ROOT, filepath)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path)
        print(f"   ✓ {filepath} ({size:,} bytes)")
    else:
        print(f"   ✗ Missing: {filepath}")
        all_ok = False

if not all_ok:
    print("\n   ERROR: Deployment package is incomplete!")
    sys.exit(1)

# Step 5: Report model size
print("\n5. Model Information:")
model_h_path = os.path.join(INCLUDE_DIR, 'model_data.h')
if os.path.exists(model_h_path):
    with open(model_h_path, 'r') as f:
        content = f.read()
        # Try to find model size
        if 'g_model_data_len' in content:
            import re
            match = re.search(r'g_model_data_len\s*=\s*(\d+)', content)
            if match:
                model_size = int(match.group(1))
                print(f"   Model size: {model_size:,} bytes ({model_size/1024:.1f} KB)")

print("\n" + "="*60)
print("✓ DEPLOYMENT PACKAGE READY!")
print("="*60)
print("\nNext steps:")
print("1. Open VSCode")
print("2. Install PlatformIO extension (if not already installed)")
print("3. Open this folder: esp32_deployment")
print("4. Connect ESP32-S3 via USB")
print("5. Click 'Upload' in PlatformIO toolbar")
print("6. Click 'Monitor' to view serial output")
print("\nFor detailed instructions, see README.md")
print("="*60)
