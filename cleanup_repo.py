import os
import shutil

def cleanup_repository():
    print("==================================================")
    print("   Repository Cleanup & Refactoring Script")
    print("==================================================")
    
    # Files to explicitly keep
    keep_files = {
        'SE_MSCNN_v2_improved.py',       # Final PyTorch Model
        'benchmark_final_model.py',      # Quick benchmark script
        'generate_detailed_documentation.py', # Heavy documentation generation
        'generate_word_report.py',       # IEEE Report generation
        'weights.v2_improved.pt',        # Final Checkpoint
        'preprocessed_data.pkl',         # Cached data
        'model_evaluation_report.docx',  # Generated IEEE Report
        'Comprehensive_20_Page_Documentation.docx', # Generated detail history
        'benchmark_plot.png',            # Generated visualization
        'cleanup_repo.py',               # This script
        'README.md',
        'LICENSE',
        'requirements.txt',
        'SE-MSCNN_v2_predictions.csv'
    }
    
    # Directories to keep (we don't delete standard dirs, just loose experimental scripts)
    keep_dirs = {
        '.git',
        '__pycache__',
        'apnea-ecg-database-1.0.0',
        'dataset',
        'models',
        'output',
        'pic',
        'utils',
        'code_baseline',
        'code_modern'
    }

    deleted_count = 0
    
    for item in os.listdir('.'):
        if os.path.isfile(item):
            # Don't delete dotfiles like .gitignore
            if item.startswith('.'):
                continue
                
            # If it's an old experiment/notebook/weights file, trash it
            if item not in keep_files:
                if item.endswith('.py') or item.endswith('.csv') or item.endswith('.txt') or item.endswith('.keras') or item.endswith('.md') or item.endswith('.pkl') or item.endswith('.ipynb'):
                    print(f"Deleting obsolete artifact: {item}")
                    try:
                        os.remove(item)
                        deleted_count += 1
                    except Exception as e:
                        print(f"  Error deleting {item}: {e}")
                        
        elif os.path.isdir(item):
            # We don't indiscriminately delete dirs unless we explicitly want to
            pass

    print(f"\nCleanup complete. Removed {deleted_count} obsolete files.")
    print("The repository is now fully refactored and contains only the final V2 pipeline.")

if __name__ == '__main__':
    cleanup_repository()
