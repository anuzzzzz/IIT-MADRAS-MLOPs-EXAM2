import sys
import os

def test_phase2():
    print("=" * 50)
    print("PHASE 2 TESTING: EXPLAINABILITY + FAIRNESS")
    print("=" * 50)
    
    try:
        # Test explainability
        print("1. Testing explainability analysis...")
        sys.path.append('src')
        os.chdir('src')
        exec(open('explainability.py').read())
        print("✅ Explainability analysis completed")
        
        # Test fairness
        print("\n2. Testing fairness analysis...")  
        exec(open('fairness_testing.py').read())
        print("✅ Fairness analysis completed")
        
        os.chdir('..')  # Back to main directory
        
        print("\n" + "=" * 50)
        print("PHASE 2 COMPLETED SUCCESSFULLY! ✅")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_phase2()
