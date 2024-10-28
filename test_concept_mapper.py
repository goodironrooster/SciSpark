# test_concept_mapper.py

from concept_mapper_tester import ConceptMapperTester

def main():
    print("Starting Concept Mapper Testing Suite...")
    print("=" * 50)
    
    # Initialize tester with verbose output
    tester = ConceptMapperTester()
    tester.verbose = True
    
    try:
        # Run all tests
        tester.run_all_tests()
        
        # Print detailed results
        tester.print_results()
        
    except Exception as e:
        print(f"Critical testing error: {str(e)}")
        raise

if __name__ == "__main__":
    main()