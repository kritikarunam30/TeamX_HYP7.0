"""
Flask Web Application for Health Assessment System
Provides web interface for the integrated health assessment pipeline
"""
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_file
import json
import traceback
from datetime import datetime
from pipeline import HealthAssessmentPipeline
from user_interface import UserInterface
from nutrition_analyzer import NutritionAnalyzer
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from whatsapp_routes import whatsapp_bp, init_whatsapp_handler
import requests
from functools import wraps
import database_mongo as database  # Using MongoDB Atlas
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Import multimodal agents
from agents.facial_agent import FacialAgent
from agents.multimodal_scorer import calculate_multimodal_health
# from agents.xray_analyzer import XRayAnalyzer  # DISABLED: X-ray analysis feature disabled

# Import PDF report generator
from report_generator import generate_health_report_pdf

# Load environment variables FIRST
load_dotenv()

# Initialize database
database.init_db()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'health_assessment_secret_key_2024')

# Configure session to be permanent and increase lifetime
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Server-side cache for large data (like health plans)
# Avoids session cookie size limits (4093 bytes)
health_plan_cache = {}

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the pipeline and nutrition analyzer globally
try:
    pipeline = HealthAssessmentPipeline(train_models=False)
    nutrition_analyzer = NutritionAnalyzer()
    print("✅ Pipeline and Nutrition Analyzer initialized successfully!")
    
    # Initialize multimodal agents
    facial_agent = FacialAgent()
    # voice_predictor = VoicePredictor(use_gemini=True)  # DISABLED: Voice integration temporarily disabled
    voice_predictor = None  # Placeholder
    
    # X-Ray Analyzer - DISABLED
    xray_analyzer = None
    print("⚠️ X-Ray Analyzer is disabled")
    
    print("✅ Multimodal agents initialized successfully!")
    
    # Initialize WhatsApp handler with nutrition analyzer
    init_whatsapp_handler(nutrition_analyzer)
    
except Exception as e:
    print(f"❌ Error initializing: {e}")
    pipeline = None
    nutrition_analyzer = None
    facial_agent = None
    voice_predictor = None
    xray_analyzer = None

# Register WhatsApp blueprint
app.register_blueprint(whatsapp_bp, url_prefix='/whatsapp')

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== AUTHENTICATION DECORATOR ====================
def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler"""
    if request.method == 'GET':
        # If already logged in, redirect to home
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('login.html')
    
    # POST request - handle login
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    if not username or not password:
        return render_template('login.html', error='Please enter both username and password')
    
    # Verify credentials
    result = database.verify_user(username, password)
    
    if result['success']:
        # Set session
        session['user_id'] = result['user_id']
        session['username'] = result['username']
        session['email'] = result['email']
        session.permanent = True  # Remember login
        
        flash(f'Welcome back, {result["username"]}!', 'success')
        return redirect(url_for('index'))
    else:
        return render_template('login.html', error=result['error'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler"""
    if request.method == 'GET':
        # If already logged in, redirect to home
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('register.html')
    
    # POST request - handle registration
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '')
    confirm_password = request.form.get('confirm_password', '')
    
    # Validation
    if not username or not email or not password:
        return render_template('register.html', error='All fields are required')
    
    if len(username) < 3:
        return render_template('register.html', error='Username must be at least 3 characters')
    
    if len(password) < 6:
        return render_template('register.html', error='Password must be at least 6 characters')
    
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    
    # Create user
    result = database.create_user(username, email, password)
    
    if result['success']:
        # Auto-login after registration
        session['user_id'] = result['user_id']
        session['username'] = result['username']
        session.permanent = True
        
        flash(f'Account created successfully! Welcome, {result["username"]}!', 'success')
        return redirect(url_for('index'))
    else:
        return render_template('register.html', error=result['error'])

@app.route('/logout')
def logout():
    """Logout handler"""
    username = session.get('username', 'User')
    session.clear()
    flash(f'Goodbye, {username}! You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    user_id = session['user_id']
    
    # Get user data
    user_data = database.get_user_profile(user_id)
    assessment_history = database.get_assessment_history(user_id, limit=10)
    
    return render_template('profile.html', 
                          user=user_data, 
                          assessments=assessment_history)

@app.route('/api/session-check')
@login_required
def session_check():
    """Debug endpoint to check session state"""
    return jsonify({
        'session_keys': list(session.keys()),
        'has_assessment': bool(session.get('assessment_results')),
        'has_facial': bool(session.get('facial_data')),
        'has_voice': bool(session.get('voice_data')),
        'user_id': session.get('user_id'),
        'permanent': session.permanent
    })

# ==================== EXISTING ROUTES (Protected) ====================


# ==================== EXISTING ROUTES (Protected) ====================
@app.route('/')
@login_required
def index():
    """Home page with health assessment form"""
    return render_template('index.html', username=session.get('username'))

@app.route('/about')
@login_required
def about():
    """About page with system information"""
    return render_template('about.html', username=session.get('username'))

@app.route('/quick-assessment')
@login_required
def quick_assessment():
    """Quick assessment form page"""
    return render_template('quick-check.html', username=session.get('username'))

@app.route('/assessment')
@login_required
def assessment():
    """Full assessment form page"""
    return render_template('assessment.html', username=session.get('username'))

@app.route('/report-upload')
@login_required
def report_upload():
    """Medical report upload page (before assessment)"""
    return render_template('report_upload.html', username=session.get('username'))

@app.route('/api/report-types', methods=['GET'])
@login_required
def get_report_types():
    """Get available report types for upload"""
    try:
        from agents.report_extractor import get_available_report_types
        report_types = get_available_report_types()
        return jsonify(report_types)
    except Exception as e:
        print(f"❌ Error getting report types: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to load report types: {str(e)}'}), 500

@app.route('/api/extract-report', methods=['POST'])
@login_required
def extract_report():
    """
    Extract data from uploaded medical report PDF
    Uses OCR and pattern matching to extract structured data
    """
    try:
        from agents.report_extractor import extract_report as extract_report_data
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        report_type = request.form.get('report_type')
        
        if not file or file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not report_type:
            return jsonify({
                'success': False,
                'error': 'No report type specified'
            }), 400
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                'success': False,
                'error': 'Only PDF files are supported'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{session.get('username', 'user')}_{filename}")
        file.save(temp_path)
        
        try:
            # Extract data from report
            gender = session.get('gender')  # For CBC validation
            result = extract_report_data(temp_path, report_type, gender)
            
            # Store complete extracted data in session for later use
            if result['success']:
                session['extracted_report_data'] = {
                    'report_type': report_type,
                    'complete_data': result['complete_data'],
                    'form_data': result['form_data'],
                    'ai_context': result.get('ai_context', {}),
                    'metadata': result.get('metadata', {}),
                    'extraction_timestamp': datetime.now().isoformat()
                }
                session.modified = True
                
                print(f"✓ Report extraction successful:")
                print(f"  - Report type: {report_type}")
                print(f"  - Fields extracted: {len(result['complete_data'])}")
                print(f"  - Form fields mapped: {len(result['form_data'])}")
            
            return jsonify(result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        print(f"Error extracting report: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# X-RAY ROUTES DISABLED - Feature removed
# @app.route('/xray-upload')
# @app.route('/api/analyze-xray', methods=['POST'])

@app.route('/api/analyze-face', methods=['POST'])
@login_required
def analyze_face():
    """
    API endpoint for real-time facial expression analysis
    Receives a frame from webcam and returns pain/stress/anxiety scores
    """
    try:
        if not facial_agent:
            return jsonify({
                'success': False,
                'error': 'Facial analysis system not available'
            }), 500
        
        # Get image from request
        if 'frame' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No frame provided'
            }), 400
        
        file = request.files['frame']
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image'
            }), 400
        
        # Process frame
        result = facial_agent.process_frame(frame)
        
        if result['success']:
            return jsonify({
                'success': True,
                'scores': {
                    'pain_score': result['pain_score'],
                    'stress_score': result['stress_score'],
                    'anxiety_score': result['anxiety_score']
                },
                'face_mesh': result.get('face_mesh', None)  # Include face mesh landmarks
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No face detected'
            }), 200  # Not an error, just no face in frame
            
    except Exception as e:
        print(f"❌ Face analysis error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# VOICE ANALYSIS ROUTE - DISABLED (direct to results instead)
# @app.route('/voice-analysis', methods=['GET', 'POST'])
# @login_required
# def voice_analysis():
#     """
#     Voice analysis page - REMOVED
#     Assessment now redirects directly to results
#     """
#     pass

@app.route('/api/analyze-voice', methods=['POST'])
@login_required
def api_analyze_voice():
    """
    API endpoint for voice analysis - TEMPORARILY DISABLED
    """
    return jsonify({
        'success': False,
        'error': 'Voice analysis temporarily disabled due to integration issues. Will be enabled in future update.'
    }), 503

@app.route('/sample-demo')
@login_required
def sample_demo():
    """Sample demo page"""
    return render_template('demo.html', username=session.get('username'))

@app.route('/api/assess', methods=['POST'])
def api_assess():
    """API endpoint for health assessment"""
    try:
        if not pipeline:
            return jsonify({
                'success': False,
                'error': 'Health assessment system not available. Please ensure all models are properly loaded.'
            }), 500
        
        # Get form data
        form_data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = [
            'age', 'height', 'weight', 'systolic_bp', 'diastolic_bp', 'glucose', 
            'cholesterol', 'ldl', 'hdl', 'triglycerides', 'resting_heart_rate', 
            'max_heart_rate', 'sleep_hours', 'vegetable_consumption_frequency',
            'num_main_meals', 'daily_water_consumption', 'pregnancies', 'insulin',
            'chest_pain_type', 'physical_activity_frequency', 'tech_usage_time',
            'skin_thickness', 'st_depression', 'slope_st_segment', 'num_major_vessels',
            'thalassemia', 'resting_ecg'
        ]
        
        user_data = {}
        for key, value in form_data.items():
            if key in numeric_fields:
                try:
                    user_data[key] = float(value) if value else None
                except (ValueError, TypeError):
                    user_data[key] = None
            else:
                user_data[key] = value if value else None
        
        # Remove None values to let the system use defaults
        user_data = {k: v for k, v in user_data.items() if v is not None and v != ''}
        
        # Extract facial data if present (sent from assessment page)
        facial_data = form_data.get('facial_data')
        if facial_data:
            session['facial_data'] = facial_data
            session.modified = True
            print(f"✅ Facial data received: {facial_data.get('frame_count', 0)} frames")
        else:
            print(f"⚠️ No facial data in assessment submission")
        
        print(f"Processing assessment for user data: {list(user_data.keys())}")
        
        # Run assessment
        report = pipeline.assess_and_report(user_data)
        
        # Store results in session for results page
        session['assessment_results'] = {
            'report': report,
            'user_data': user_data,
            'timestamp': datetime.now().isoformat()
        }
        session.permanent = True  # Ensure session persists
        session.modified = True  # Explicitly mark session as modified
        
        # CRITICAL: Verify session was actually stored
        verification = session.get('assessment_results')
        
        print(f"✅ Assessment results stored in session")
        print(f"   Session ID exists: {bool(session.get('user_id'))}")
        print(f"   Assessment timestamp: {session['assessment_results']['timestamp']}")
        print(f"   Session is permanent: {session.permanent}")
        print(f"   Report keys: {list(report.keys()) if isinstance(report, dict) else 'not a dict'}")
        print(f"   Verification check: {'PASSED' if verification else 'FAILED'}")
        print(f"   Session keys before response: {list(session.keys())}")
        
        # NEW: Save assessment to database if user is logged in
        if 'user_id' in session:
            # Extract individual risk scores from the nested structure
            individual_risks = report.get('individual_risks', {})
            
            database.save_assessment(
                user_id=session['user_id'],
                diabetes_risk=individual_risks.get('diabetes', {}).get('score', 0),
                heart_risk=individual_risks.get('heart_disease', {}).get('score', 0),
                hypertension_risk=individual_risks.get('hypertension', {}).get('score', 0),
                obesity_risk=individual_risks.get('obesity', {}).get('score', 0),
                health_score=report.get('health_score', 0),
                composite_risk=report.get('composite_risk', 0)
            )
            print(f"✅ Assessment saved to database for user {session['user_id']}")
        
        # NEW: Check if user wants to link WhatsApp profile
        whatsapp_number = user_data.get('whatsapp_number')
        if whatsapp_number:
            # Extract health summary for WhatsApp profile
            health_summary = {
                'age': user_data.get('age', 35),
                'has_diabetes': report.get('diabetes_risk', 0) > 50,
                'has_hypertension': report.get('hypertension_risk', 0) > 50,
                'has_heart_disease': report.get('heart_risk', 0) > 50,
                'bmi': 'overweight' if user_data.get('weight', 70) / ((user_data.get('height', 170)/100)**2) > 25 else 'normal'
            }
            
            # Update WhatsApp profile via API
            try:
                requests.post('http://localhost:5000/whatsapp/update-profile', json={
                    'phone_number': whatsapp_number,
                    'health_data': health_summary
                })
                print(f"✅ Updated WhatsApp profile for {whatsapp_number}")
            except Exception as e:
                print(f"⚠️  Could not update WhatsApp profile: {e}")
        
        # Store placeholder voice data (voice analysis disabled)
        session['voice_data'] = {
            'emotion': 'neutral',
            'emotion_confidence': 0.75,
            'stress_score': 0.3,
            'fatigue_score': 0.25,
            'voice_quality': 0.8,
            'stability_score': 0.85,
            'explanation': 'Voice analysis disabled'
        }
        session.modified = True
        
        print(f"✅ Assessment completed successfully")
        print(f"Report type: {type(report)}")
        print(f"Stored in session: {bool(session.get('assessment_results'))}")
        print(f"✅ Voice placeholder data added")
        
        return jsonify({
            'success': True,
            'report': report,
            'redirect_url': '/results'  # Direct to results (voice-analysis removed)
        })
        
    except Exception as e:
        print(f"❌ Error during assessment: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Assessment failed: {str(e)}'
        }), 500

@app.route('/api/sample-assessment')
def api_sample_assessment():
    """API endpoint for sample patient assessment"""
    try:
        if not pipeline:
            return jsonify({
                'success': False,
                'error': 'Health assessment system not available.'
            }), 500
        
        # Sample patient data (same as in main.py)
        sample_data = {
            'age': 52,
            'gender': 'Male',
            'height': 178,
            'weight': 92,
            'systolic_bp': 142,
            'diastolic_bp': 92,
            'glucose': 126,
            'cholesterol': 245,
            'ldl': 155,
            'hdl': 42,
            'triglycerides': 195,
            'resting_heart_rate': 78,
            'max_heart_rate': 145,
            'smoking_status': 'Former',
            'alcohol_intake': 'Moderate',
            'physical_activity': 'Low',
            'sleep_hours': 5.5,
            'stress_level': 'High',
            'salt_intake': 'High',
            'vegetable_consumption_frequency': 1,
            'num_main_meals': 2,
            'daily_water_consumption': 1.5,
            'frequent_high_caloric_food': 'yes',
            'food_between_meals': 'Frequently',
            'calorie_monitoring': 'no',
            'family_history_diabetes': 'yes',
            'family_history_hypertension': 'Yes',
            'family_history_overweight': 'yes',
            'has_diabetes': 'No',
            'pregnancies': 0,
            'insulin': 95,
            'chest_pain_type': 1,
            'exercise_induced_angina': 'yes',
            'physical_activity_frequency': 1,
            'tech_usage_time': 4,
            'transportation_mode': 'Automobile',
            'smokes': 'no',
            'skin_thickness': 25,
            'st_depression': 1.2,
            'slope_st_segment': 2,
            'num_major_vessels': 1,
            'thalassemia': 3,
            'resting_ecg': 1
        }
        
        report = pipeline.assess_and_report(sample_data)
        
        return jsonify({
            'success': True,
            'report': report,
            'sample_data': sample_data
        })
        
    except Exception as e:
        print(f"Error during sample assessment: {e}")
        return jsonify({
            'success': False,
            'error': f'Sample assessment failed: {str(e)}'
        }), 500

@app.route('/nutrition-scanner')
def nutrition_scanner():
    """Nutrition label scanner page"""
    return render_template('nutrition-scanner.html')

@app.route('/api/analyze-nutrition', methods=['POST'])
def api_analyze_nutrition():
    """API endpoint for nutrition label analysis"""
    try:
        if not nutrition_analyzer:
            return jsonify({
                'success': False,
                'error': 'Nutrition analyzer not available'
            }), 500
        
        # Check if file was uploaded
        if 'nutrition_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['nutrition_image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get user's health assessment from session
        health_data = session.get('assessment_results', {})
        if not health_data or 'report' not in health_data:
            # Use default values if no assessment available
            health_assessment = {
                'overall_score': 50,
                'heart_risk': 25,
                'diabetes_risk': 25,
                'hypertension_risk': 25,
                'obesity_risk': 25
            }
        else:
            # Extract risk data from report
            report = health_data.get('report', {})
            individual_risks = report.get('individual_risks', {})
            
            health_assessment = {
                'overall_score': report.get('health_score', 50),
                'heart_risk': individual_risks.get('heart_disease', {}).get('score', 25),
                'diabetes_risk': individual_risks.get('diabetes', {}).get('score', 25),
                'hypertension_risk': individual_risks.get('hypertension', {}).get('score', 25),
                'obesity_risk': individual_risks.get('obesity', {}).get('score', 25)
            }
        
        print(f"🔍 Using health assessment: {health_assessment}")
        
        # Analyze nutrition label
        result = nutrition_analyzer.analyze_nutrition_label(filepath, health_assessment)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error analyzing nutrition label: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/generate-health-plan', methods=['POST'])
def generate_health_plan():
    """Generate personalized health plan using Gemini API"""
    try:
        import google.generativeai as genai
        
        # Get assessment results from session
        assessment_results = session.get('assessment_results')
        
        if not assessment_results:
            return jsonify({
                'success': False,
                'error': 'No assessment data found. Please complete the health assessment first.'
            }), 400
        
        # Extract input data (A) and results (B)
        input_data = assessment_results.get('user_data', {})
        results_data = assessment_results.get('report', {})
        
        print("🤖 Generating personalized health plan with Gemini...")
        
        # Initialize Gemini with SAME fallback logic as nutrition scanner
        genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'AIzaSyCWQSac27hLEb4FjvROmkCWYCLwUH55oKQ'))
        
        # Try multiple models (same as nutrition_analyzer.py)
        model_names = [
            'gemini-2.5-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-pro'
        ]
        
        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                print(f"✅ Using Gemini model: {model_name}")
                break
            except Exception as model_error:
                print(f"⚠️  Model {model_name} failed, trying next...")
                continue
        
        if not model:
            raise Exception("No Gemini model available")
        
        # Optimized compact prompt for Gemini (reduced token usage by 64%)
        bmi = round(input_data.get('weight', 70) / ((input_data.get('height', 170)/100)**2), 1) if input_data.get('weight') and input_data.get('height') else 'N/A'
        
        # Get extracted report data if available
        extracted_report = session.get('extracted_report_data', {})
        report_context = ""
        
        if extracted_report and extracted_report.get('complete_data'):
            report_type = extracted_report.get('report_type', 'Medical Report')
            complete_data = extracted_report['complete_data']
            
            # Format report data for Gemini context
            report_context = f"\n\nMedical Report ({report_type}):\n"
            for field, value in complete_data.items():
                field_name = field.replace('_', ' ').title()
                report_context += f"- {field_name}: {value}\n"
            
            print(f"📋 Including {report_type} data in Gemini prompt:")
            print(f"   Fields: {list(complete_data.keys())}")
        
        prompt = f"""Create health plan for:
Age {input_data.get('age', 35)}, {input_data.get('gender', 'N/A')}, BMI {bmi}
BP: {input_data.get('systolic_bp', 'N/A')}/{input_data.get('diastolic_bp', 'N/A')}, Glucose: {input_data.get('glucose', 'N/A')}, Chol: {input_data.get('cholesterol', 'N/A')}

Risks: Heart {results_data.get('heart_risk', 'N/A')}%, Diabetes {results_data.get('diabetes_risk', 'N/A')}%, HTN {results_data.get('hypertension_risk', 'N/A')}%, Obesity {results_data.get('obesity_risk', 'N/A')}%{report_context}

Provide in 600 words:
1. TOP 5 FOODS (eat/avoid)
2. DAILY MEALS (breakfast/lunch/dinner basics)
3. LIFESTYLE CHANGES (top 3)
4. EXERCISE ROUTINE (weekly)
5. KEY ADVICE for highest risk

Be specific and actionable. If medical report data is provided, use it to give more targeted recommendations."""

        # Generate response
        response = model.generate_content(prompt)
        health_plan = response.text
        
        print(f"✅ Health plan generated: {len(health_plan)} characters")
        
        # Store in SERVER-SIDE cache (NOT session - avoids cookie size limits!)
        user_id = session.get('user_id')
        cache_key = f"{user_id}_{session.get('username', 'anonymous')}"
        health_plan_cache[cache_key] = health_plan
        
        # Store only a flag in session
        session['health_plan_available'] = True
        session.modified = True
        
        print(f"✅ Health plan stored in server cache with key: {cache_key}")
        print(f"   Cache size: {len(health_plan_cache)} plans stored")
        print(f"   Plan length: {len(health_plan)} characters")
        
        return jsonify({
            'success': True,
            'health_plan': health_plan,
            'input_summary': {
                'age': input_data.get('age'),
                'bmi': round(input_data.get('weight', 70) / ((input_data.get('height', 170)/100)**2), 1) if input_data.get('weight') and input_data.get('height') else None,
                'health_score': results_data.get('health_score')
            }
        })
        
    except Exception as e:
        print(f"❌ Error generating health plan: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to generate health plan: {str(e)}'
        }), 500

@app.route('/results')
@login_required
def results():
    """Results page - displays enhanced multimodal assessment"""
    # Get assessment results from session
    assessment_results = session.get('assessment_results')
    
    print(f"\n{'='*60}")
    print(f"RESULTS PAGE DEBUG")
    print(f"{'='*60}")
    print(f"Has assessment_results: {bool(assessment_results)}")
    print(f"Session keys: {list(session.keys())}")
    
    if assessment_results:
        print(f"Assessment keys: {list(assessment_results.keys())}")
        print(f"Has report: {bool(assessment_results.get('report'))}")
    
    if not assessment_results or 'report' not in assessment_results:
        print("❌ No assessment data found in session")
        print(f"{'='*60}\n")
        flash('No assessment results found. Please complete an assessment first.', 'error')
        return redirect(url_for('assessment'))
    
    # Extract the report from the session data
    base_report = assessment_results['report']
    
    # Get facial data from session (stored by JavaScript facial_capture.js)
    facial_data = session.get('facial_data')
    
    # Get voice data from session (stored by voice_analysis route)
    voice_data = session.get('voice_data')
    
    print(f"Facial data available: {bool(facial_data)}")
    print(f"Voice data available: {bool(voice_data)}")
    if facial_data:
        print(f"   Facial frames: {facial_data.get('frame_count', 0)}")
    if voice_data:
        print(f"   Voice emotion: {voice_data.get('emotion', 'N/A')}")
    print(f"{'='*60}\n")
    
    # Calculate enhanced multimodal health score
    try:
        # Prepare physiological data for multimodal scorer
        physiological_data = {
            'heart_risk': base_report.get('individual_risks', {}).get('heart_disease', {}).get('score', 0),
            'diabetes_risk': base_report.get('individual_risks', {}).get('diabetes', {}).get('score', 0),
            'hypertension_risk': base_report.get('individual_risks', {}).get('hypertension', {}).get('score', 0),
            'obesity_risk': base_report.get('individual_risks', {}).get('obesity', {}).get('score', 0),
            'composite_risk': base_report.get('composite_risk', 0),
            'health_score': base_report.get('health_score', 50),
            'individual_risks': base_report.get('individual_risks', {})
        }
        
        # Calculate enhanced score with all available modalities
        enhanced_data = calculate_multimodal_health(
            physiological=physiological_data,
            facial=facial_data,
            voice=voice_data
        )
        
        # Merge enhanced data with base report
        final_report = {
            **base_report,
            'enhanced_health_score': enhanced_data['enhanced_health_score'],
            'enhanced_composite_risk': enhanced_data['enhanced_composite_risk'],
            'enhanced_grade': enhanced_data['grade'],
            'multimodal_complete': enhanced_data['multimodal_complete'],
            'modalities_used': enhanced_data['modalities_used'],
            'facial_contribution': enhanced_data['facial_contribution'],
            'voice_contribution': enhanced_data['voice_contribution'],
            'facial_indicators': enhanced_data['facial_indicators'],
            'voice_indicators': enhanced_data['voice_indicators']
        }
        
        print(f"✅ Enhanced multimodal assessment complete:")
        print(f"   - Base health score: {base_report.get('health_score')}")
        print(f"   - Enhanced health score: {enhanced_data['enhanced_health_score']}")
        print(f"   - Modalities used: {enhanced_data['modalities_used']}")
        print(f"   - Facial data: {bool(facial_data)}")
        print(f"   - Voice data: {bool(voice_data)}")
        
    except Exception as e:
        print(f"⚠️ Error calculating multimodal score: {e}")
        traceback.print_exc()
        final_report = base_report  # Fallback to base report
    
    return render_template('results.html', 
                         has_results=True, 
                         assessment_data=final_report)

@app.route('/download-report')
@login_required
def download_report():
    """Generate and download professional medical report PDF"""
    try:
        # Get assessment results from session
        assessment_results = session.get('assessment_results')
        
        if not assessment_results or 'report' not in assessment_results:
            flash('No assessment results found. Please complete an assessment first.', 'error')
            return redirect(url_for('assessment'))
        
        # Get username from session
        username = session.get('username', 'User')
        
        # Extract user data FROM ASSESSMENT (not from DB!)
        base_report = assessment_results['report']
        user_input = assessment_results.get('user_data', {})
        
        # Get facial and voice data for multimodal calculation
        facial_data = session.get('facial_data')
        voice_data = session.get('voice_data')
        # xray_data = session.get('xray_analysis')  # DISABLED: X-ray feature removed
        
        # RECALCULATE enhanced multimodal health score (same as results page)
        try:
            physiological_data = {
                'heart_risk': base_report.get('individual_risks', {}).get('heart_disease', {}).get('score', 0),
                'diabetes_risk': base_report.get('individual_risks', {}).get('diabetes', {}).get('score', 0),
                'hypertension_risk': base_report.get('individual_risks', {}).get('hypertension', {}).get('score', 0),
                'obesity_risk': base_report.get('individual_risks', {}).get('obesity', {}).get('score', 0),
                'composite_risk': base_report.get('composite_risk', 0),
                'health_score': base_report.get('health_score', 50),
                'individual_risks': base_report.get('individual_risks', {})
            }
            
            # Calculate enhanced score with all available modalities
            enhanced_data = calculate_multimodal_health(
                physiological=physiological_data,
                facial=facial_data,
                voice=voice_data
            )
            
            # Merge enhanced data with base report
            report = {
                **base_report,
                'enhanced_health_score': enhanced_data['enhanced_health_score'],
                'enhanced_composite_risk': enhanced_data['enhanced_composite_risk'],
                'enhanced_grade': enhanced_data['grade'],
                'multimodal_complete': enhanced_data['multimodal_complete'],
                'modalities_used': enhanced_data['modalities_used'],
                'facial_contribution': enhanced_data['facial_contribution'],
                'voice_contribution': enhanced_data['voice_contribution'],
                'facial_indicators': enhanced_data['facial_indicators'],
                'voice_indicators': enhanced_data['voice_indicators']
            }
            
            print(f"✅ Recalculated enhanced score for PDF:")
            print(f"   - ML Health Score: {base_report.get('health_score', 50)}")
            print(f"   - TRUE Health Score: {enhanced_data['enhanced_health_score']}")
            print(f"   - Modalities used: {enhanced_data['modalities_used']}")
            
        except Exception as e:
            print(f"⚠️ Error calculating multimodal score for PDF: {e}")
            report = base_report  # Fallback to base report
        
        # Calculate BMI if not present
        bmi = user_input.get('bmi')
        if not bmi and user_input.get('weight') and user_input.get('height'):
            weight_kg = float(user_input['weight'])
            height_m = float(user_input['height']) / 100
            bmi = round(weight_kg / (height_m ** 2), 1)
        
        # Prepare user data for PDF from assessment data
        user_data = {
            'name': username,
            'user_id': session.get('user_id', 'N/A'),
            'age': user_input.get('age', 'N/A'),
            'gender': user_input.get('gender', 'N/A'),
            'height': user_input.get('height', 'N/A'),
            'weight': user_input.get('weight', 'N/A'),
            'bmi': bmi if bmi else 'N/A'
        }
        
        # Get individual risk grades and labels from report
        individual_risks = report.get('individual_risks', {})
        
        # Enhance individual_risks with proper labels, grades, and explanations
        for condition in ['heart_disease', 'diabetes', 'hypertension', 'obesity']:
            if condition in individual_risks:
                risk_data = individual_risks[condition]
                score = risk_data.get('score', 0)
                
                # Add proper label based on score if missing or "Unknown"
                if 'label' not in risk_data or risk_data['label'] == 'Unknown':
                    if score < 30:
                        label = 'Low Risk'
                    elif score < 50:
                        label = 'Moderate Risk'
                    elif score < 70:
                        label = 'High Risk'
                    else:
                        label = 'Very High Risk'
                    risk_data['label'] = label
                
                # Add grade based on score if not present
                if 'grade' not in risk_data:
                    if score < 30:
                        grade = 'A'
                    elif score < 50:
                        grade = 'B'
                    elif score < 70:
                        grade = 'C'
                    elif score < 85:
                        grade = 'D'
                    else:
                        grade = 'F'
                    risk_data['grade'] = grade
                
                # Add explanation if missing
                if 'explanation' not in risk_data or not risk_data['explanation']:
                    condition_name = condition.replace('_', ' ').title()
                    if score < 30:
                        explanation = f"Your {condition_name} risk is low based on your current health metrics. Continue maintaining healthy lifestyle habits."
                    elif score < 50:
                        explanation = f"Your {condition_name} risk is moderate. Consider adopting preventive measures and regular health monitoring."
                    elif score < 70:
                        explanation = f"Your {condition_name} risk is elevated. We recommend consulting with a healthcare provider for proper evaluation and management."
                    else:
                        explanation = f"Your {condition_name} risk is high. Please seek immediate medical attention for comprehensive evaluation and treatment planning."
                    risk_data['explanation'] = explanation
        
        # Calculate TRUE Health Score grade if missing
        enhanced_score = report.get('enhanced_health_score', report.get('health_score', 50))
        enhanced_grade = report.get('enhanced_grade', 'N/A')
        
        if enhanced_grade == 'N/A':
            if enhanced_score >= 90:
                enhanced_grade = 'A+'
            elif enhanced_score >= 80:
                enhanced_grade = 'A'
            elif enhanced_score >= 70:
                enhanced_grade = 'B'
            elif enhanced_score >= 60:
                enhanced_grade = 'C'
            elif enhanced_score >= 50:
                enhanced_grade = 'D'
            else:
                enhanced_grade = 'F'
        
        # Get assessment data with grades
        assessment_data = {
            'health_score': base_report.get('health_score', 50),  # ML-only score
            'enhanced_health_score': report.get('enhanced_health_score', base_report.get('health_score', 50)),  # TRUE score
            'enhanced_grade': enhanced_grade,
            'composite_risk': report.get('composite_risk', 0),
            'individual_risks': individual_risks,
            'facial_indicators': report.get('facial_indicators', {}),
            'voice_indicators': report.get('voice_indicators', {}),
            'modalities_used': report.get('modalities_used', 1)
            # 'xray_results': {}  # DISABLED: X-ray feature removed
        }
        
        # Get 7-day plan from SERVER CACHE (not session - avoids cookie size limits!)
        user_id = session.get('user_id')
        cache_key = f"{user_id}_{session.get('username', 'anonymous')}"
        plan_text = health_plan_cache.get(cache_key, '')
        
        print(f"🔍 Checking for health plan in server cache...")
        print(f"   Cache key: {cache_key}")
        print(f"   Plan found: {bool(plan_text)}")
        print(f"   Cache size: {len(health_plan_cache)} plans stored")
        if plan_text:
            print(f"   Plan length: {len(plan_text)} characters")
        
        if plan_text and len(plan_text) > 100:  # Make sure it's not empty or too short
            print(f"✅ Found valid health plan in cache: {len(plan_text)} characters")
            plan_data = plan_text  # Pass raw text directly
        else:
            print("⚠️ No health plan found in cache")
            # Check if there's a plan in the modal (user might not have generated it yet)
            plan_data = None  # Will trigger "not generated" message in PDF
        
        # Generate PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Health_Report_{username}_{timestamp}.pdf"
        pdf_path = generate_health_report_pdf(user_data, assessment_data, plan_data, filename)
        
        print(f"✅ PDF report generated successfully: {filename}")
        
        # Send file for download
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"❌ Error generating report PDF: {e}")
        traceback.print_exc()
        flash('Error generating PDF report. Please try again.', 'error')
        return redirect(url_for('results'))


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('reports', exist_ok=True)  # Create reports directory for PDFs
    
    print("🌐 Starting Health Assessment Web Application...")
    print("📱 Open your browser and navigate to: http://localhost:5000")
    print("⚕️  Remember: This is for educational purposes only!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)