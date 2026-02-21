"""
Professional Medical Report PDF Generator - COMPLETE REWRITE
Generates hospital-grade comprehensive health assessment reports with 7-day personalized plans
Version 3.0 - Enhanced Professional Formatting
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.graphics.shapes import Drawing, Rect, String, Circle, Line, Polygon
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.spider import SpiderChart
from reportlab.graphics import renderPDF
from datetime import datetime, timedelta
import os
import re
import math


# Professional Color Palette (Hospital Medical Theme)
class MedicalColors:
    """Professional medical color scheme"""
    PRIMARY_BLUE = colors.HexColor('#0047AB')  # Medical blue
    DARK_BLUE = colors.HexColor('#001F3F')      # Navy
    LIGHT_BLUE = colors.HexColor('#E8F4F8')     # Light blue background
    SUCCESS_GREEN = colors.HexColor('#00A86B')  # Medical green
    WARNING_ORANGE = colors.HexColor('#FF8C00')  # Warning
    DANGER_RED = colors.HexColor('#DC143C')      # Alert red
    NEUTRAL_GRAY = colors.HexColor('#708090')    # Slate gray
    LIGHT_GRAY = colors.HexColor('#F5F5F5')      # Background
    TEXT_BLACK = colors.HexColor('#212529')      # Primary text
    TEXT_GRAY = colors.HexColor('#6C757D')       # Secondary text
    BORDER_GRAY = colors.HexColor('#DEE2E6')     # Borders
    WHITE = colors.white


class MedicalReportGenerator:
    """
    Generate hospital-grade medical PDF reports with professional formatting
    
    Features:
    - Professional medical color scheme
    - Consistent typography and spacing
    - Clear data visualization tables
    - Proper medical disclaimer
    - Page numbering and headers
    - Multi-page support with consistent styling
    """
    
    def __init__(self):
        """Initialize report generator with custom medical styles"""
        self.styles = getSampleStyleSheet()
        self.colors = MedicalColors()
        self._create_professional_styles()
    
    def _create_professional_styles(self):
        """Create professional medical document styles"""
        
        # Helper function to add style only if it doesn't exist
        def add_style_if_not_exists(style):
            if style.name not in self.styles:
                self.styles.add(style)
        
        # Main Report Title
        add_style_if_not_exists(ParagraphStyle(
            name='MedicalTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=self.colors.PRIMARY_BLUE,
            alignment=TA_CENTER,
            spaceAfter=8,
            spaceBefore=0,
            leading=26
        ))
        
        # Report Subtitle
        add_style_if_not_exists(ParagraphStyle(
            name='MedicalSubtitle',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            textColor=self.colors.TEXT_GRAY,
            alignment=TA_CENTER,
            spaceAfter=16,
            spaceBefore=0
        ))
        
        # Section Headers (e.g., PATIENT INFORMATION, EXECUTIVE SUMMARY)
        add_style_if_not_exists(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=self.colors.DARK_BLUE,
            spaceAfter=10,
            spaceBefore=16,
            leading=18,
            borderPadding=8,
            backColor=self.colors.LIGHT_BLUE,
            leftIndent=10
        ))
        
        # Subsection Headers (e.g., Risk Assessment Overview)
        add_style_if_not_exists(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=self.colors.TEXT_BLACK,
            spaceAfter=8,
            spaceBefore=12,
            leading=14
        ))
        
        # Body Text
        add_style_if_not_exists(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=self.colors.TEXT_BLACK,
            alignment=TA_LEFT,
            spaceAfter=6,
            leading=14
        ))
        
        # Body Text Justified
        add_style_if_not_exists(ParagraphStyle(
            name='BodyTextJustified',
            parent=self.styles['BodyText'],
            alignment=TA_JUSTIFY
        ))
        
        # Bullet Points
        add_style_if_not_exists(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['BodyText'],
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=4
        ))
        
        # Score Display (Large Numbers)
        add_style_if_not_exists(ParagraphStyle(
            name='ScoreDisplay',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=32,
            textColor=self.colors.PRIMARY_BLUE,
            alignment=TA_CENTER,
            spaceAfter=4,
            leading=36
        ))
        
        # Alert/Warning Text
        add_style_if_not_exists(ParagraphStyle(
            name='AlertText',
            parent=self.styles['BodyText'],
            fontName='Helvetica-Bold',
            textColor=self.colors.DANGER_RED,
            fontSize=10
        ))
        
        # Footer/Disclaimer Text
        add_style_if_not_exists(ParagraphStyle(
            name='FooterText',
            parent=self.styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=self.colors.TEXT_GRAY,
            alignment=TA_JUSTIFY,
            leading=10
        ))
        
        # Small Caption Text
        add_style_if_not_exists(ParagraphStyle(
            name='CaptionText',
            parent=self.styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            textColor=self.colors.TEXT_GRAY,
            alignment=TA_CENTER,
            spaceAfter=0
        ))
        
        # Add aliases for backward compatibility
        add_style_if_not_exists(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['SectionHeader'],
        ))
        
        add_style_if_not_exists(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['SubsectionHeader'],
        ))
        
        add_style_if_not_exists(ParagraphStyle(
            name='MedicalBodyText',
            parent=self.styles['BodyText'],
        ))
    
    def generate_report(self, user_data, assessment_data, plan_data, output_path):
        """
        Generate complete professional medical report PDF
        
        Args:
            user_data (dict): Patient information (name, age, gender, height, weight, etc.)
            assessment_data (dict): Health assessment results from ML models
            plan_data (str/dict): 7-day health plan (Gemini-generated or structured)
            output_path (str): Full path where PDF will be saved
            
        Returns:
            str: Path to generated PDF file
        """
        
        # Create PDF document with professional margins
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.6*inch,
            bottomMargin=0.75*inch,
            title="Health Assessment Report",
            author="HealthVision AI System"
        )
        
        # Build content elements
        story = []
        
        # PAGE 1: Header + Patient Info + Executive Summary
        story.extend(self._build_header())
        story.extend(self._build_patient_info(user_data))
        story.append(Spacer(1, 0.2*inch))
        story.extend(self._build_executive_summary(assessment_data))
        
        # PAGE 2: Detailed Risk Assessment
        story.append(PageBreak())
        story.extend(self._build_detailed_results(assessment_data))
        
        # PAGE 3: Multimodal Analysis (Facial + X-Ray if available)
        story.append(PageBreak())
        story.extend(self._build_multimodal_analysis(assessment_data))
        
        # PAGE 4+: 7-Day Health Plan
        story.append(PageBreak())
        story.extend(self._build_health_plan(plan_data))
        
        # FINAL PAGE: Recommendations + Disclaimer
        story.append(PageBreak())
        story.extend(self._build_recommendations(assessment_data))
        story.extend(self._build_footer())
        
        # Build PDF with custom page template
        doc.build(story, 
                  onFirstPage=self._add_page_number, 
                  onLaterPages=self._add_page_number)
        
        return output_path
    
    def _build_header(self):
        """Build professional report header with title and date"""
        elements = []
        
        # Main Title
        elements.append(Paragraph(
            "COMPREHENSIVE HEALTH ASSESSMENT REPORT",
            self.styles['MedicalTitle']
        ))
        
        # Subtitle with Generation Date
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        elements.append(Paragraph(
            f"Report Generated: {report_date}",
            self.styles['MedicalSubtitle']
        ))
        
        # Horizontal divider
        elements.append(HRFlowable(
            width="100%",
            thickness=2,
            color=self.colors.PRIMARY_BLUE,
            spaceBefore=8,
            spaceAfter=16
        ))
        
        return elements
    
    def _build_patient_info(self, user_data):
        """Build patient information section with professional table"""
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "PATIENT INFORMATION",
            self.styles['SectionHeader']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        # Calculate BMI if not provided
        bmi = user_data.get('bmi')
        if not bmi:
            try:
                height_m = float(user_data.get('height', 170)) / 100
                weight_kg = float(user_data.get('weight', 70))
                bmi = round(weight_kg / (height_m ** 2), 1)
            except:
                bmi = 'N/A'
        
        # Generate Report ID
        report_id = f"HR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Patient Information Table (2 columns x 5 rows)
        patient_data = [
            ['Patient Name:', user_data.get('name', 'Not Provided'), 
             'Report ID:', report_id],
            ['Age:', f"{user_data.get('age', 'N/A')} years", 
             'Gender:', user_data.get('gender', 'Not Specified').title()],
            ['Height:', f"{user_data.get('height', 'N/A')} cm", 
             'Weight:', f"{user_data.get('weight', 'N/A')} kg"],
            ['Body Mass Index (BMI):', f"{bmi}", 
             'Assessment Date:', datetime.now().strftime("%Y-%m-%d")],
        ]
        
        patient_table = Table(
            patient_data, 
            colWidths=[1.6*inch, 1.9*inch, 1.6*inch, 1.9*inch],
            rowHeights=[0.35*inch] * len(patient_data)
        )
        
        patient_table.setStyle(TableStyle([
            # Header columns (labels) - Light blue background
            ('BACKGROUND', (0, 0), (0, -1), self.colors.LIGHT_BLUE),
            ('BACKGROUND', (2, 0), (2, -1), self.colors.LIGHT_BLUE),
            
            # Font styles
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTNAME', (3, 0), (3, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            
            # Text colors
            ('TEXTCOLOR', (0, 0), (0, -1), self.colors.DARK_BLUE),
            ('TEXTCOLOR', (2, 0), (2, -1), self.colors.DARK_BLUE),
            ('TEXTCOLOR', (1, 0), (-1, -1), self.colors.TEXT_BLACK),
            
            # Alignment
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors.BORDER_GRAY),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, self.colors.PRIMARY_BLUE),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(patient_table)
        
        return elements
    
    def _build_executive_summary(self, assessment_data):
        """Build executive summary with health scores and risk overview"""
        elements = []
        
        # Section Header
        elements.append(Paragraph(
            "EXECUTIVE SUMMARY",
            self.styles['SectionHeader']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        # Extract scores safely - check both possible keys
        ml_score = float(assessment_data.get('health_score', assessment_data.get('ml_health_score', 50)))
        enhanced_score = float(assessment_data.get('enhanced_health_score', 
                                                   assessment_data.get('true_health_score', ml_score)))
        grade = assessment_data.get('enhanced_grade', assessment_data.get('overall_grade', 'N/A'))
        
        # Health Score Display Box with consistent font sizes
        score_box_data = [
            [
                Paragraph('<b>ML Health Score</b><br/><font size="9">(Physiological Data Only)</font>', 
                         self.styles['BodyText']),
                Paragraph(f'<font size="24" color="{self.colors.PRIMARY_BLUE}"><b>{ml_score:.1f}</b></font><br/>'
                         f'<font size="10">out of 100</font>', 
                         self.styles['BodyText'])
            ],
            [
                Paragraph('<b>TRUE Health Score™</b><br/><font size="9">(Multimodal Enhanced)</font>', 
                         self.styles['BodyText']),
                Paragraph(f'<font size="24" color="{self.colors.SUCCESS_GREEN}"><b>{enhanced_score:.1f}</b></font><br/>'
                         f'<font size="12"><b>Grade: {grade}</b></font>', 
                         self.styles['BodyText'])
            ]
        ]
        
        score_box = Table(score_box_data, colWidths=[2.2*inch, 4.8*inch])
        score_box.setStyle(TableStyle([
            # Background colors
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#EBF5FF')),
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#E6F7F1')),
            
            # Borders
            ('BOX', (0, 0), (-1, -1), 2, self.colors.PRIMARY_BLUE),
            ('LINEBELOW', (0, 0), (-1, 0), 1, self.colors.BORDER_GRAY),
            ('GRID', (0, 0), (-1, -1), 1, self.colors.BORDER_GRAY),
            
            # Alignment
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(score_box)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add visual health score comparison (progress bars like profile page)
        elements.extend(self._create_health_score_comparison(ml_score, enhanced_score))
        elements.append(Spacer(1, 0.15*inch))
        
        # Risk Assessment Overview
        elements.append(Paragraph(
            "Risk Assessment Overview",
            self.styles['SubsectionHeader']
        ))
        elements.append(Spacer(1, 0.08*inch))
        
        # Build risk summary table
        individual_risks = assessment_data.get('individual_risks', {})
        
        risk_table_data = [
            ['Disease Risk', 'Risk Level', 'Grade', 'Score', 'Status']
        ]
        
        # Define risk categories with proper labels
        risk_categories = [
            ('heart_disease', 'Cardiovascular Disease'),
            ('diabetes', 'Type 2 Diabetes'),
            ('hypertension', 'Hypertension (High BP)'),
            ('obesity', 'Obesity')
        ]
        
        for category_key, category_label in risk_categories:
            risk_info = individual_risks.get(category_key, {})
            
            # Safely extract values
            risk_score = float(risk_info.get('score', 0))
            risk_label = str(risk_info.get('label', 'Unknown'))
            risk_grade = str(risk_info.get('grade', 'N/A'))
            
            # Determine status color and icon
            if risk_score < 30:
                status_color = self.colors.SUCCESS_GREEN
                status_text = '✓ Low Risk'
                status_bg = colors.HexColor('#E6F7F1')
            elif risk_score < 60:
                status_color = self.colors.WARNING_ORANGE
                status_text = '⚠ Moderate'
                status_bg = colors.HexColor('#FFF4E6')
            else:
                status_color = self.colors.DANGER_RED
                status_text = '⚠ High Risk'
                status_bg = colors.HexColor('#FFEEF0')
            
            risk_table_data.append([
                category_label,
                risk_label,
                risk_grade,
                f"{risk_score:.1f}%",
                Paragraph(f'<font color="{status_color}"><b>{status_text}</b></font>', 
                         self.styles['BodyText'])
            ])
        
        risk_table = Table(
            risk_table_data, 
            colWidths=[1.8*inch, 1.3*inch, 0.7*inch, 0.9*inch, 1.3*inch]
        )
        risk_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), self.colors.DARK_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors.WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            
            # Data rows
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Borders
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors.BORDER_GRAY),
            ('LINEBELOW', (0, 0), (-1, 0), 2, self.colors.PRIMARY_BLUE),
            
            # Padding
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(risk_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Add visual risk progress bars (matching profile page)
        elements.extend(self._create_risk_progress_bars(individual_risks))
        
        # Add star diagram visualization
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("HEALTH RISK VISUALIZATION", self.styles['SubsectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            "The radar chart below provides a comprehensive visual overview of your health risk profile across all assessed categories. "
            "Larger areas indicate higher risk levels requiring attention.",
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 0.15*inch))
        
        # Center the star diagram
        star_diagram = self._create_star_diagram(individual_risks)
        elements.append(star_diagram)
        elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_detailed_results(self, assessment_data):
        """Build detailed assessment results"""
        elements = []
        
        elements.append(Paragraph("DETAILED HEALTH ASSESSMENT", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        individual_risks = assessment_data.get('individual_risks', {})
        
        # Heart Disease Analysis
        if 'heart_disease' in individual_risks:
            elements.extend(self._build_condition_analysis('Heart Disease', 
                                                          individual_risks['heart_disease']))
        
        # Diabetes Analysis
        if 'diabetes' in individual_risks:
            elements.extend(self._build_condition_analysis('Diabetes', 
                                                          individual_risks['diabetes']))
        
        # Hypertension Analysis
        if 'hypertension' in individual_risks:
            elements.extend(self._build_condition_analysis('Hypertension', 
                                                          individual_risks['hypertension']))
        
        # Obesity Analysis
        if 'obesity' in individual_risks:
            elements.extend(self._build_condition_analysis('Obesity', 
                                                          individual_risks['obesity']))
        
        return elements
    
    def _build_condition_analysis(self, condition_name, condition_data):
        """Build analysis section for a specific condition"""
        elements = []
        
        elements.append(Paragraph(f"● {condition_name}", self.styles['SubsectionHeading']))
        
        score = condition_data.get('score', 0)
        label = condition_data.get('label', 'Unknown')
        explanation = condition_data.get('explanation', 'No detailed analysis available.')
        
        # Score and label
        elements.append(Paragraph(
            f'<b>Risk Score:</b> {score:.1f}% - <b>{label}</b>',
            self.styles['MedicalBodyText']
        ))
        
        # Explanation
        elements.append(Paragraph(
            f'<b>Analysis:</b> {explanation}',
            self.styles['MedicalBodyText']
        ))
        
        elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _build_multimodal_analysis(self, assessment_data):
        """Build multimodal analysis section (facial + voice)"""
        elements = []
        
        elements.append(Paragraph("MULTIMODAL HEALTH ANALYSIS", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph(
            "This advanced assessment incorporates multiple data modalities for comprehensive health evaluation:",
            self.styles['MedicalBodyText']
        ))
        elements.append(Spacer(1, 0.1*inch))
        
        # Modalities used
        modalities = assessment_data.get('modalities_used', 1)
        elements.append(Paragraph(f"<b>Modalities Analyzed:</b> {modalities}/3", self.styles['MedicalBodyText']))
        
        # Weight distribution
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("Assessment Weight Distribution", self.styles['SubsectionHeading']))
        
        weight_data = [
            ['Modality', 'Weight', 'Contribution'],
            ['Physiological Data (ML Models)', '60%', f"{assessment_data.get('health_score', 50):.1f}/100"],
            ['Facial Expression Analysis', '30%', 'Included' if assessment_data.get('facial_indicators') else 'Not Available'],
            ['Voice Analysis', '10%', 'Placeholder']
        ]
        
        weight_table = Table(weight_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        weight_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b82f6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(weight_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Facial indicators
        facial_indicators = assessment_data.get('facial_indicators', {})
        if facial_indicators:
            elements.append(Paragraph("Facial Expression Analysis", self.styles['SubsectionHeading']))
            
            facial_data = [
                ['Indicator', 'Score (0-10)', 'Interpretation'],
                ['Pain Indicators', 
                 f"{facial_indicators.get('avg_pain_score', 0):.1f}", 
                 self._interpret_facial_score(facial_indicators.get('avg_pain_score', 0))],
                ['Stress Levels', 
                 f"{facial_indicators.get('avg_stress_score', 0):.1f}", 
                 self._interpret_facial_score(facial_indicators.get('avg_stress_score', 0))],
                ['Anxiety Indicators', 
                 f"{facial_indicators.get('avg_anxiety_score', 0):.1f}", 
                 self._interpret_facial_score(facial_indicators.get('avg_anxiety_score', 0))],
            ]
            
            facial_table = Table(facial_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            facial_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#a855f7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            elements.append(facial_table)
            
            frame_count = facial_indicators.get('frame_count', 0)
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(
                f"<i>Analysis based on {frame_count} facial expression frames captured during assessment.</i>",
                self.styles['MedicalBodyText']
            ))
        
        return elements
    
    def _interpret_facial_score(self, score):
        """Interpret facial analysis score"""
        if score < 3:
            return "Minimal/Normal"
        elif score < 6:
            return "Moderate Levels"
        else:
            return "Elevated Levels"
    
    def _build_health_plan(self, plan_data):
        """Build 7-day health plan section"""
        elements = []
        
        elements.append(Paragraph("PERSONALIZED 7-DAY HEALTH PLAN", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Check if plan was generated
        if not plan_data or (isinstance(plan_data, str) and len(plan_data) < 100):
            # No plan generated yet
            elements.append(Paragraph(
                "<b>⚠️ Health Plan Not Generated</b>",
                self.styles['AlertText']
            ))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(
                "A personalized health plan has not been generated yet. To get your customized 7-day health plan:",
                self.styles['MedicalBodyText']
            ))
            elements.append(Spacer(1, 0.05*inch))
            elements.append(Paragraph(
                "1. Return to the Results page<br/>"
                "2. Click the <b>'Get Personalized Health Plan'</b> button<br/>"
                "3. Wait for the AI-powered plan to be generated<br/>"
                "4. Download a new report with your complete health plan included",
                self.styles['MedicalBodyText']
            ))
            return elements
        
        # Check if plan_data is a string (Gemini-generated text) or structured data
        if isinstance(plan_data, str):
            # Handle raw text from Gemini
            elements.append(Paragraph(
                "A comprehensive plan tailored to your specific health profile and risk factors.",
                self.styles['MedicalBodyText']
            ))
            elements.append(Spacer(1, 0.15*inch))
            
            # Split the plan text into paragraphs for better formatting
            plan_paragraphs = plan_data.split('\n\n')
            for para in plan_paragraphs:
                if para.strip():
                    # Check if it's a heading (starts with #, **, or numbers)
                    if para.strip().startswith('#') or para.strip().startswith('**') or (len(para.strip()) > 1 and para.strip()[0].isdigit() and para.strip()[1] in '.):'):
                        # Format as subheading
                        clean_text = para.strip().replace('#', '').replace('**', '').strip()
                        elements.append(Paragraph(clean_text, self.styles['SubsectionHeading']))
                    else:
                        # Regular paragraph
                        # Convert markdown bold to HTML (handle nested **)
                        formatted_text = para.strip()
                        while '**' in formatted_text:
                            formatted_text = formatted_text.replace('**', '<b>', 1).replace('**', '</b>', 1)
                        # Handle bullet points
                        if formatted_text.startswith('- ') or formatted_text.startswith('* '):
                            formatted_text = '• ' + formatted_text[2:]
                        elements.append(Paragraph(formatted_text, self.styles['MedicalBodyText']))
                    elements.append(Spacer(1, 0.1*inch))
        
        else:
            # Handle structured dict data (legacy format)
            elements.append(Paragraph(
                "A comprehensive 7-day plan tailored to your specific health profile and risk factors.",
                self.styles['MedicalBodyText']
            ))
            elements.append(Spacer(1, 0.15*inch))
            
            # Iterate through each day
            for day_num in range(1, 8):
                day_key = f'day_{day_num}'
                day_data = plan_data.get(day_key, {})
                
                if not day_data:
                    continue
                
                # Day header
                day_date = (datetime.now() + timedelta(days=day_num-1)).strftime("%A, %B %d")
                elements.append(Paragraph(
                    f"DAY {day_num} - {day_date}",
                    self.styles['SubsectionHeading']
                ))
                
                # Day focus
                focus = day_data.get('focus', 'General Wellness')
                elements.append(Paragraph(f"<b>Focus:</b> {focus}", self.styles['MedicalBodyText']))
                
                # Create table for daily activities
                activities = []
                
                # Morning
                if 'morning' in day_data:
                    activities.append(['🌅 Morning', day_data['morning']])
                
                # Afternoon
                if 'afternoon' in day_data:
                    activities.append(['☀️ Afternoon', day_data['afternoon']])
                
                # Evening
                if 'evening' in day_data:
                    activities.append(['🌙 Evening', day_data['evening']])
                
                # Meals
                if 'meals' in day_data:
                    activities.append(['🍽️ Nutrition', day_data['meals']])
                
                # Exercise
                if 'exercise' in day_data:
                    activities.append(['💪 Exercise', day_data['exercise']])
                
                if activities:
                    day_table = Table(activities, colWidths=[1.5*inch, 4.5*inch])
                    day_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eff6ff')),
                        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1e40af')),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('LEFTPADDING', (0, 0), (-1, -1), 10),
                    ]))
                    
                    elements.append(day_table)
                
                elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _build_recommendations(self, assessment_data):
        """Build recommendations section"""
        elements = []
        
        elements.append(Paragraph("MEDICAL RECOMMENDATIONS", self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        # General recommendations
        elements.append(Paragraph("General Health Recommendations:", self.styles['SubsectionHeading']))
        
        recommendations = [
            "Schedule regular follow-up appointments with your healthcare provider",
            "Monitor vital signs (blood pressure, glucose levels) as recommended",
            "Maintain a balanced diet rich in fruits, vegetables, and whole grains",
            "Engage in regular physical activity (at least 150 minutes per week)",
            "Prioritize adequate sleep (7-9 hours per night)",
            "Manage stress through mindfulness, meditation, or relaxation techniques",
            "Stay hydrated with adequate water intake throughout the day",
            "Avoid smoking and limit alcohol consumption"
        ]
        
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", self.styles['MedicalBodyText']))
        
        elements.append(Spacer(1, 0.15*inch))
        
        # Specific recommendations based on risk factors
        individual_risks = assessment_data.get('individual_risks', {})
        high_risk_conditions = []
        
        for condition, data in individual_risks.items():
            if data.get('score', 0) >= 60:
                high_risk_conditions.append(condition.replace('_', ' ').title())
        
        if high_risk_conditions:
            elements.append(Paragraph("⚠ Priority Attention Required:", self.styles['SubsectionHeading']))
            elements.append(Paragraph(
                f"<font color='#dc2626'><b>Elevated risk detected for: {', '.join(high_risk_conditions)}</b></font>",
                self.styles['AlertText']
            ))
            elements.append(Paragraph(
                "We strongly recommend consulting with a healthcare professional for further evaluation and personalized medical advice.",
                self.styles['MedicalBodyText']
            ))
        
        return elements
    
    def _build_footer(self):
        """Build report footer"""
        elements = []
        
        elements.append(Spacer(1, 0.3*inch))
        elements.append(self._create_line())
        elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Paragraph(
            "<i>DISCLAIMER: This report is generated based on AI-powered health assessment models and is intended for "
            "informational purposes only. It should not be considered as professional medical advice, diagnosis, or treatment. "
            "Always seek the advice of qualified healthcare providers with any questions regarding your medical condition.</i>",
            self.styles['MedicalBodyText']
        ))
        
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"<b>System Version:</b> 2.0 (Multimodal) | "
            f"<b>Confidential Medical Document</b>",
            ParagraphStyle(
                name='Footer',
                parent=self.styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#64748b'),
                alignment=TA_CENTER
            )
        ))
        
        return elements
    
    def _create_progress_bar(self, value, max_value=100, width=400, height=25):
        """Create a horizontal progress bar (like in the profile page)"""
        drawing = Drawing(width + 10, height + 10)
        
        # Calculate bar width
        percentage = (value / max_value) * 100
        bar_fill_width = (percentage / 100) * width
        
        # Determine color based on value
        if percentage < 30:
            fill_color = self.colors.SUCCESS_GREEN
        elif percentage < 60:
            fill_color = self.colors.WARNING_ORANGE
        else:
            fill_color = self.colors.DANGER_RED
        
        # Background bar (gray)
        bg_bar = Rect(5, 5, width, height)
        bg_bar.fillColor = self.colors.LIGHT_GRAY
        bg_bar.strokeColor = self.colors.BORDER_GRAY
        bg_bar.strokeWidth = 1
        drawing.add(bg_bar)
        
        # Filled bar (colored based on risk)
        if bar_fill_width > 0:
            fill_bar = Rect(5, 5, bar_fill_width, height)
            fill_bar.fillColor = fill_color
            fill_bar.strokeColor = None
            drawing.add(fill_bar)
        
        # Percentage text overlay
        text = String(width / 2 + 5, height / 2, f'{value:.1f}%',
                     fontSize=11, fillColor=colors.white if percentage > 50 else self.colors.TEXT_BLACK,
                     textAnchor='middle', fontName='Helvetica-Bold')
        drawing.add(text)
        
        return drawing
    
    def _create_risk_progress_bars(self, individual_risks):
        """Create progress bars for each disease risk (matching profile page style)"""
        elements = []
        
        risk_categories = [
            ('heart_disease', 'Cardiovascular Disease Risk'),
            ('diabetes', 'Type 2 Diabetes Risk'),
            ('hypertension', 'Hypertension Risk'),
            ('obesity', 'Obesity Risk')
        ]
        
        for category_key, category_label in risk_categories:
            risk_info = individual_risks.get(category_key, {})
            risk_score = float(risk_info.get('score', 0))
            risk_label = str(risk_info.get('label', 'Unknown'))
            risk_grade = str(risk_info.get('grade', 'N/A'))
            
            # Add label
            elements.append(Paragraph(
                f'<b>{category_label}</b> • {risk_label} (Grade: {risk_grade})',
                self.styles['BodyText']
            ))
            elements.append(Spacer(1, 0.05*inch))
            
            # Add progress bar
            elements.append(self._create_progress_bar(risk_score))
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _create_health_score_comparison(self, ml_score, enhanced_score):
        """Create side-by-side progress bars comparing ML vs TRUE scores"""
        elements = []
        
        # ML Score
        elements.append(Paragraph(
            '<b>ML Health Score</b> <font size="9">(Physiological Data Only)</font>',
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 0.05*inch))
        elements.append(self._create_progress_bar(ml_score))
        elements.append(Spacer(1, 0.15*inch))
        
        # TRUE Score
        elements.append(Paragraph(
            '<b>TRUE Health Score™</b> <font size="9">(Multimodal AI Enhanced)</font>',
            self.styles['BodyText']
        ))
        elements.append(Spacer(1, 0.05*inch))
        elements.append(self._create_progress_bar(enhanced_score))
        elements.append(Spacer(1, 0.15*inch))
        
        return elements
    
    def _create_star_diagram(self, individual_risks):
        """Create a star/radar diagram showing health risk levels with axis labels"""
        # Create drawing with extra space for labels
        drawing = Drawing(450, 450)
        
        # Get risk scores
        heart_risk = float(individual_risks.get('heart_disease', {}).get('score', 0))
        diabetes_risk = float(individual_risks.get('diabetes', {}).get('score', 0))
        hypertension_risk = float(individual_risks.get('hypertension', {}).get('score', 0))
        obesity_risk = float(individual_risks.get('obesity', {}).get('score', 0))
        
        # Use ReportLab's SpiderChart for professional radar diagram
        chart = SpiderChart()
        chart.width = 320
        chart.height = 320
        chart.x = 65
        chart.y = 65
        
        # Set up the spider chart
        chart.data = [[heart_risk, diabetes_risk, hypertension_risk, obesity_risk]]
        chart.labels = ['Heart\nDisease', 'Diabetes', 'Hypertension', 'Obesity']
        
        # Styling
        chart.strands[0].fillColor = colors.HexColor('#0047AB40')  # Semi-transparent blue
        chart.strands[0].strokeColor = self.colors.PRIMARY_BLUE
        chart.strands[0].strokeWidth = 2.5
        
        # Spider web styling
        chart.spokes.strokeColor = self.colors.BORDER_GRAY
        chart.spokes.strokeWidth = 1.5
        chart.spokeLabels.fontName = 'Helvetica-Bold'
        chart.spokeLabels.fontSize = 9
        chart.spokeLabels.fillColor = self.colors.TEXT_BLACK
        
        # Web circles (concentric rings)
        chart.strandLabels.fontName = 'Helvetica'
        chart.strandLabels.fontSize = 7
        chart.strandLabels.fillColor = self.colors.TEXT_GRAY
        
        drawing.add(chart)
        
        # Add axis scale labels (0%, 25%, 50%, 75%, 100%)
        center_x = 225
        center_y = 225
        
        # Add tiny scale markers on the left side (vertical axis)
        scale_values = [0, 25, 50, 75, 100]
        for i, val in enumerate(scale_values):
            y_pos = center_y - 160 + (i * 80)  # Spread from bottom to top
            # Left side scale
            scale_label = String(center_x - 170, y_pos, f'{val}%',
                               fontSize=7, fillColor=self.colors.TEXT_GRAY,
                               textAnchor='end', fontName='Helvetica')
            drawing.add(scale_label)
        
        # Add axis titles
        # Y-axis label (vertical - left side)
        y_label = String(15, center_y, 'Risk Level (%)',
                        fontSize=8, fillColor=self.colors.TEXT_BLACK,
                        textAnchor='middle', fontName='Helvetica-Bold')
        # Rotate for vertical text
        y_label_rotated = y_label
        drawing.add(y_label_rotated)
        
        # X-axis label (horizontal - bottom)
        x_label = String(center_y, 15, 'Health Risk Assessment',
                        fontSize=8, fillColor=self.colors.TEXT_BLACK,
                        textAnchor='middle', fontName='Helvetica-Bold')
        drawing.add(x_label)
        
        # Add legend box in top-right corner
        legend_x = 340
        legend_y = 410
        
        # Legend background
        legend_bg = Rect(legend_x, legend_y - 35, 95, 40)
        legend_bg.fillColor = colors.HexColor('#F8F9FA')
        legend_bg.strokeColor = self.colors.BORDER_GRAY
        legend_bg.strokeWidth = 0.5
        drawing.add(legend_bg)
        
        # Legend title
        legend_title = String(legend_x + 47.5, legend_y - 10, 'Risk Zones',
                            fontSize=7, fillColor=self.colors.TEXT_BLACK,
                            textAnchor='middle', fontName='Helvetica-Bold')
        drawing.add(legend_title)
        
        # Legend items
        legend_items = [
            ('Low', self.colors.SUCCESS_GREEN, legend_y - 18),
            ('Med', self.colors.WARNING_ORANGE, legend_y - 26),
            ('High', self.colors.DANGER_RED, legend_y - 34)
        ]
        
        for label, color, y_pos in legend_items:
            # Color box
            color_box = Rect(legend_x + 5, y_pos, 8, 6)
            color_box.fillColor = color
            color_box.strokeColor = None
            drawing.add(color_box)
            
            # Label text
            label_text = String(legend_x + 16, y_pos + 1, f'{label} Risk',
                              fontSize=6, fillColor=self.colors.TEXT_GRAY,
                              textAnchor='start', fontName='Helvetica')
            drawing.add(label_text)
        
        return drawing
    
    def _create_line(self):
        """Create horizontal line"""
        return Table([['']], colWidths=[6.5*inch], rowHeights=[2],
                    style=TableStyle([
                        ('LINEABOVE', (0, 0), (-1, 0), 2, colors.HexColor('#cbd5e1'))
                    ]))
    
    def _add_page_number(self, canvas, doc):
        """Add page numbers to each page"""
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawRightString(7.5*inch, 0.5*inch, text)
        canvas.restoreState()


def generate_health_report_pdf(user_data, assessment_data, plan_data, output_filename=None):
    """
    Convenience function to generate health report PDF
    
    Args:
        user_data: User information dictionary
        assessment_data: Health assessment results dictionary
        plan_data: 7-day health plan dictionary
        output_filename: Optional custom filename
        
    Returns:
        Path to generated PDF file
    """
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_id = user_data.get('user_id', 'unknown')
        output_filename = f"health_report_{user_id}_{timestamp}.pdf"
    
    # Ensure output directory exists
    output_dir = os.path.join('reports')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Generate report
    generator = MedicalReportGenerator()
    generator.generate_report(user_data, assessment_data, plan_data, output_path)
    
    print(f"✅ Medical report generated: {output_path}")
    
    return output_path
