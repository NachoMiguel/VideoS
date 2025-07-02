# AI Video Slicer - Test Suite

This directory contains all tests for the AI Video Slicer project, organized into a comprehensive test suite that validates all functionality.

## Test Structure

### Unit Tests
- **`test_core.py`** - Tests for core functionality (config, exceptions, logging)
- **`test_services.py`** - Tests for service modules (OpenAI, ElevenLabs, Image Search)
- **`test_video.py`** - Tests for video processing components (face detection, video processor)

### Integration Tests
- **`test_implementations.py`** - Comprehensive integration tests that verify all implementations work together

### Test Runner
- **`run_all_tests.py`** - Master test runner that executes all test suites and provides detailed reporting

## Running Tests

### Run All Tests
```bash
cd backend/tests
python run_all_tests.py
```

### Run Individual Test Suites
```bash
# Core functionality tests
python test_core.py

# Service module tests
python test_services.py

# Video processing tests
python test_video.py

# Integration tests
python test_implementations.py
```

### Run Specific Test Classes
```bash
# Run specific test class
python -m unittest test_core.TestCoreConfig

# Run specific test method
python -m unittest test_core.TestCoreConfig.test_settings_initialization
```

## Test Coverage

### Core Functionality (test_core.py)
- ✅ Configuration settings validation
- ✅ File size limits verification
- ✅ Exception handling
- ✅ Logging functionality

### Services (test_services.py)
- ✅ OpenAI character extraction
- ✅ Script segmentation
- ✅ ElevenLabs voice selection
- ✅ API key rotation
- ✅ Image search functionality
- ✅ Character image retrieval
- ✅ Filename sanitization

### Video Processing (test_video.py)
- ✅ Face detection initialization
- ✅ Basic and advanced face detection
- ✅ Scene transition detection methods
- ✅ Frame difference calculation
- ✅ Histogram analysis
- ✅ Luminosity change detection
- ✅ Edge density analysis
- ✅ Quality score calculation
- ✅ Scene duration limiting
- ✅ Transition creation

### Integration Tests (test_implementations.py)
- ✅ Character extraction from scripts
- ✅ Character image search
- ✅ Face detection system
- ✅ Audio generation pipeline
- ✅ Video processor enhancements
- ✅ Script analysis functionality

## Test Requirements

### Dependencies
All test dependencies are included in the main project requirements. Key testing libraries:
- `unittest` (Python standard library)
- `asyncio` (for async test support)
- `numpy` and `cv2` (for video processing tests)
- `tempfile` (for temporary file handling)

### Environment Setup
Tests use the same environment configuration as the main application. Make sure you have:
- Python 3.8+
- All dependencies from `requirements.txt` installed
- Proper API keys configured (for integration tests)

## Test Data

Tests use minimal synthetic data to avoid dependencies on external files:
- Generated test images for face detection
- Sample scripts for character extraction
- Mock data structures for video processing

## Continuous Integration

The test suite is designed to be CI/CD friendly:
- No external file dependencies
- Comprehensive error handling
- Detailed reporting
- Exit codes for automated systems

## Adding New Tests

When adding new functionality, please:

1. **Add unit tests** to the appropriate test file
2. **Add integration tests** if the feature involves multiple components
3. **Update this README** if new test categories are added
4. **Run the full test suite** to ensure no regressions

### Test Naming Conventions
- Test files: `test_<module>.py`
- Test classes: `Test<Component>`
- Test methods: `test_<functionality>`

### Test Structure
```python
def test_functionality_name(self):
    """Test description explaining what is being tested."""
    # Arrange
    # Act
    # Assert
```

## Troubleshooting

### Common Issues

**Import Errors**: Make sure you're running tests from the correct directory and the backend path is properly configured.

**API Key Errors**: Integration tests may fail if API keys are not configured. This is expected in development environments.

**OpenCV Errors**: Video processing tests require OpenCV. Install with `pip install opencv-python`.

**Timeout Issues**: Some integration tests may timeout if external services are slow. This is normal and doesn't indicate code issues.

### Test Environment
Tests are designed to work in various environments:
- Local development
- CI/CD pipelines
- Docker containers
- Different operating systems

## Performance

The test suite is optimized for speed:
- Unit tests run in milliseconds
- Integration tests may take several seconds
- Full suite typically completes in under 60 seconds

## Contributing

When contributing tests:
1. Follow existing patterns and conventions
2. Add comprehensive docstrings
3. Test both success and failure cases
4. Include edge case testing
5. Ensure tests are deterministic and repeatable 