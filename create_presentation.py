"""
Create PDF presentation for Credit Risk Intelligence Platform
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.frames import Frame
from reportlab.lib.colors import HexColor

class NumberedPageTemplate(PageTemplate):
    def __init__(self, id, frames, **kw):
        super().__init__(id, frames, **kw)
    
    def beforeDrawPage(self, canvas, doc):
        # Draw dark blue background
        canvas.setFillColor(HexColor('#1a1a2e'))
        canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
        
        # Draw slide number
        canvas.setFillColor(HexColor('#00D4FF'))
        canvas.setFont('Helvetica', 12)
        canvas.drawRightString(doc.pagesize[0] - 72, 36, f"Slide {doc.page}")

def create_presentation():
    # Create PDF document
    doc = SimpleDocTemplate(
        "documents/presentation.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Add custom page template
    frame = Frame(72, 72, A4[0] - 144, A4[1] - 144, showBoundary=0)
    template = NumberedPageTemplate('numbered', [frame])
    doc.addPageTemplates([template])
    
    # Container for the PDF elements
    elements = []
    
    # Define custom styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=32,
        textColor=HexColor('#00D4FF'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica-Bold'
    )
    
    # Subtitle style
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=HexColor('#FFFFFF'),
        alignment=TA_CENTER,
        spaceAfter=15,
        fontName='Helvetica'
    )
    
    # Header style
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading2'],
        fontSize=26,
        textColor=HexColor('#00D4FF'),
        spaceAfter=20,
        fontName='Helvetica-Bold'
    )
    
    # Content style
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=16,
        textColor=HexColor('#FFFFFF'),
        spaceAfter=12,
        leading=22,
        fontName='Helvetica'
    )
    
    # Bullet style
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=16,
        textColor=HexColor('#FFFFFF'),
        leftIndent=30,
        spaceAfter=12,
        leading=22,
        fontName='Helvetica'
    )
    
    # Slide 1: Title
    elements.append(Paragraph("AI-Powered Credit Risk Intelligence Platform", title_style))
    elements.append(Spacer(1, 0.4*inch))
    elements.append(Paragraph("Built with LightGBM, SHAP, Groq API, Streamlit", subtitle_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("By: SK Saleena Grace", subtitle_style))
    elements.append(PageBreak())
    
    # Slide 2: Business Problem
    elements.append(Paragraph("Business Problem", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Banks need faster, accurate, explainable credit decisions", content_style))
    elements.append(Paragraph("Challenge: High loan default rates cost billions", content_style))
    elements.append(Paragraph("Solution: AI-powered risk assessment platform", content_style))
    elements.append(PageBreak())
    
    # Slide 3: Solution Architecture
    elements.append(Paragraph("Solution Architecture", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("4 Core Modules:", content_style))
    elements.append(Paragraph("1. EDA Dashboard - Data exploration and insights", bullet_style))
    elements.append(Paragraph("2. Risk Prediction - LightGBM ML model", bullet_style))
    elements.append(Paragraph("3. SHAP Explainability - Why the model decided", bullet_style))
    elements.append(Paragraph("4. Talk-to-Data Chatbot - Natural language queries", bullet_style))
    elements.append(PageBreak())
    
    # Slide 4: EDA Key Insights
    elements.append(Paragraph("EDA Key Insights", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Overall default rate: 8.07%", bullet_style))
    elements.append(Paragraph("Young applicants (<30) default more", bullet_style))
    elements.append(Paragraph("Males default at 10.14% vs females 7.00%", bullet_style))
    elements.append(Paragraph("Academic degree holders: only 1.83% default", bullet_style))
    elements.append(Paragraph("Income strongly predicts repayment ability", bullet_style))
    elements.append(PageBreak())
    
    # Slide 5: ML Model Results
    elements.append(Paragraph("ML Model Results", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Model: LightGBM with class imbalance handling", content_style))
    elements.append(Paragraph("AUC-ROC: 0.75-0.78", bullet_style))
    elements.append(Paragraph("Accuracy: 92-93%", bullet_style))
    elements.append(Paragraph("Top features: EXT_SOURCE_3, AGE_YEARS, AMT_CREDIT", bullet_style))
    elements.append(Paragraph("Risk Bands: Low/Medium/High", bullet_style))
    elements.append(PageBreak())
    
    # Slide 6: SHAP Explainability
    elements.append(Paragraph("SHAP Explainability", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("SHAP explains WHY the model made each decision", content_style))
    elements.append(Paragraph("Top features driving default risk shown visually", bullet_style))
    elements.append(Paragraph("Individual prediction explanations available", bullet_style))
    elements.append(Paragraph("Satisfies banking audit requirements", bullet_style))
    elements.append(PageBreak())
    
    # Slide 7: Talk-to-Data Chatbot
    elements.append(Paragraph("Talk-to-Data Chatbot", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Powered by Groq API (Llama 3.3-70B)", content_style))
    elements.append(Paragraph("Converts natural language to SQL automatically", bullet_style))
    elements.append(Paragraph("5 example working queries:", content_style))
    elements.append(Paragraph("- How many people defaulted?", bullet_style))
    elements.append(Paragraph("- What is average income of defaulters?", bullet_style))
    elements.append(Paragraph("- Which gender has higher default rate?", bullet_style))
    elements.append(Paragraph("- What is average loan amount?", bullet_style))
    elements.append(Paragraph("- How many applicants own a car?", bullet_style))
    elements.append(PageBreak())
    
    # Slide 8: Docker Deployment
    elements.append(Paragraph("Docker Deployment", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Fully containerized with Docker", content_style))
    elements.append(Paragraph("Single command: docker-compose up --build", bullet_style))
    elements.append(Paragraph("Runs on port 8501", bullet_style))
    elements.append(Paragraph("No manual setup required", bullet_style))
    elements.append(PageBreak())
    
    # Slide 9: Tech Stack
    elements.append(Paragraph("Tech Stack", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Python 3.10, LightGBM, SHAP", bullet_style))
    elements.append(Paragraph("Groq API - Llama 3.3-70B-Versatile (Free)", bullet_style))
    elements.append(Paragraph("Streamlit UI, SQLite Database", bullet_style))
    elements.append(Paragraph("Docker + Docker Compose", bullet_style))
    elements.append(Paragraph("Matplotlib, Seaborn, Plotly", bullet_style))
    elements.append(PageBreak())
    
    # Slide 10: Limitations and Improvements
    elements.append(Paragraph("Limitations and Improvements", header_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Current Limitations:", content_style))
    elements.append(Paragraph("- Requires Kaggle dataset manually", bullet_style))
    elements.append(Paragraph("- No user authentication", bullet_style))
    elements.append(Paragraph("- Single model (LightGBM only)", bullet_style))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Future Improvements:", content_style))
    elements.append(Paragraph("- Add XGBoost and Neural Networks", bullet_style))
    elements.append(Paragraph("- Real-time prediction API", bullet_style))
    elements.append(Paragraph("- User authentication system", bullet_style))
    
    # Build the PDF
    doc.build(elements)
    print("PDF presentation created successfully: documents/presentation.pdf")

if __name__ == "__main__":
    create_presentation()
