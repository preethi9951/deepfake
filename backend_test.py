#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Deepfake Detection System
Tests all API endpoints with various scenarios
"""

import requests
import sys
import json
import io
from datetime import datetime
from PIL import Image
import numpy as np

class DeepfakeAPITester:
    def __init__(self, base_url="https://authentic-scan.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def create_test_image(self, width=224, height=224):
        """Create a test image for upload"""
        # Create a simple test image
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes

    def test_model_status(self):
        """Test /api/model-status endpoint"""
        try:
            response = requests.get(f"{self.api_url}/model-status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'loaded' in data and 'model_path' in data and 'exists' in data:
                    if data['loaded'] and data['exists']:
                        self.log_test("Model Status - Model Loaded", True)
                        return True
                    else:
                        self.log_test("Model Status - Model Not Loaded", False, f"loaded: {data['loaded']}, exists: {data['exists']}")
                        return False
                else:
                    self.log_test("Model Status - Invalid Response Format", False, f"Missing fields in response: {data}")
                    return False
            else:
                self.log_test("Model Status - HTTP Error", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Model Status - Request Failed", False, str(e))
            return False

    def test_predict_endpoint(self):
        """Test /api/predict endpoint with image upload"""
        try:
            # Create test image
            test_image = self.create_test_image()
            
            files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['id', 'label', 'confidence', 'prediction_class', 'timestamp']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Predict - Missing Response Fields", False, f"Missing: {missing_fields}")
                    return False
                
                # Validate label
                if data['label'] not in ['REAL', 'FAKE']:
                    self.log_test("Predict - Invalid Label", False, f"Label: {data['label']}")
                    return False
                
                # Validate confidence
                if not (0 <= data['confidence'] <= 100):
                    self.log_test("Predict - Invalid Confidence", False, f"Confidence: {data['confidence']}")
                    return False
                
                # Validate prediction_class
                if data['prediction_class'] not in ['real', 'fake']:
                    self.log_test("Predict - Invalid Prediction Class", False, f"Class: {data['prediction_class']}")
                    return False
                
                self.log_test("Predict - Valid Prediction Response", True, f"Label: {data['label']}, Confidence: {data['confidence']}%")
                return True
                
            elif response.status_code == 503:
                self.log_test("Predict - Model Not Loaded", False, "Service unavailable - model not loaded")
                return False
            else:
                self.log_test("Predict - HTTP Error", False, f"Status: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Predict - Request Failed", False, str(e))
            return False

    def test_predictions_history(self):
        """Test /api/predictions endpoint"""
        try:
            response = requests.get(f"{self.api_url}/predictions?limit=10", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list):
                    self.log_test("Predictions History - Valid Response", True, f"Found {len(data)} predictions")
                    
                    # If there are predictions, validate structure
                    if len(data) > 0:
                        first_pred = data[0]
                        required_fields = ['id', 'label', 'confidence', 'prediction_class', 'timestamp']
                        missing_fields = [field for field in required_fields if field not in first_pred]
                        
                        if missing_fields:
                            self.log_test("Predictions History - Invalid Structure", False, f"Missing: {missing_fields}")
                            return False
                        else:
                            self.log_test("Predictions History - Valid Structure", True)
                    
                    return True
                else:
                    self.log_test("Predictions History - Invalid Response Type", False, f"Expected list, got: {type(data)}")
                    return False
            else:
                self.log_test("Predictions History - HTTP Error", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Predictions History - Request Failed", False, str(e))
            return False

    def test_statistics(self):
        """Test /api/stats endpoint"""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['total_predictions', 'fake_detected', 'real_detected', 'fake_percentage']
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    self.log_test("Statistics - Missing Fields", False, f"Missing: {missing_fields}")
                    return False
                
                # Validate data types and ranges
                if not isinstance(data['total_predictions'], int) or data['total_predictions'] < 0:
                    self.log_test("Statistics - Invalid Total", False, f"Total: {data['total_predictions']}")
                    return False
                
                if not isinstance(data['fake_detected'], int) or data['fake_detected'] < 0:
                    self.log_test("Statistics - Invalid Fake Count", False, f"Fake: {data['fake_detected']}")
                    return False
                
                if not isinstance(data['real_detected'], int) or data['real_detected'] < 0:
                    self.log_test("Statistics - Invalid Real Count", False, f"Real: {data['real_detected']}")
                    return False
                
                if not (0 <= data['fake_percentage'] <= 100):
                    self.log_test("Statistics - Invalid Percentage", False, f"Percentage: {data['fake_percentage']}")
                    return False
                
                # Check if totals add up
                if data['fake_detected'] + data['real_detected'] != data['total_predictions']:
                    self.log_test("Statistics - Totals Don't Match", False, 
                                f"Fake({data['fake_detected']}) + Real({data['real_detected']}) != Total({data['total_predictions']})")
                    return False
                
                self.log_test("Statistics - Valid Response", True, 
                            f"Total: {data['total_predictions']}, Fake: {data['fake_detected']}, Real: {data['real_detected']}")
                return True
            else:
                self.log_test("Statistics - HTTP Error", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Statistics - Request Failed", False, str(e))
            return False

    def test_invalid_file_upload(self):
        """Test prediction with invalid file type"""
        try:
            # Create a text file instead of image
            text_content = "This is not an image file"
            files = {'file': ('test.txt', text_content, 'text/plain')}
            
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=10)
            
            if response.status_code in [400, 422, 500]:
                self.log_test("Invalid File Upload - Proper Error Handling", True, f"Status: {response.status_code}")
                return True
            else:
                self.log_test("Invalid File Upload - No Error Handling", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Invalid File Upload - Request Failed", False, str(e))
            return False

    def test_api_root(self):
        """Test /api/ root endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'status' in data:
                    self.log_test("API Root - Valid Response", True)
                    return True
                else:
                    self.log_test("API Root - Invalid Response", False, f"Response: {data}")
                    return False
            else:
                self.log_test("API Root - HTTP Error", False, f"Status: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("API Root - Request Failed", False, str(e))
            return False

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🚀 Starting Deepfake Detection Backend API Tests")
        print("=" * 60)
        print(f"Testing API at: {self.api_url}")
        print()
        
        # Test API availability first
        self.test_api_root()
        
        # Test model status
        model_loaded = self.test_model_status()
        
        # Only run prediction tests if model is loaded
        if model_loaded:
            self.test_predict_endpoint()
            self.test_invalid_file_upload()
        else:
            print("⚠️  Skipping prediction tests - model not loaded")
        
        # Test data endpoints
        self.test_predictions_history()
        self.test_statistics()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1

def main():
    tester = DeepfakeAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())