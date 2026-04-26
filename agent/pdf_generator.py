from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

def generate_surgical_report(output_path, global_summary, actions_data):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    story = []
    
    story.append(Paragraph("Surgical Action & Segmentation Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("Global Summary of Actions", heading_style))
    story.append(Paragraph(global_summary, normal_style))
    story.append(Spacer(1, 0.5 * inch))
    
    for action in actions_data:
        action_elements = []
        
        action_elements.append(Paragraph(f"Timestamp: {action['timestamp']} - {action['action']}", heading_style))
        action_elements.append(Spacer(1, 0.1 * inch))
        
        action_elements.append(Image(action['keyframe_path'], width=6*inch, height=4*inch))
        action_elements.append(Spacer(1, 0.1 * inch))
        
        action_elements.append(Paragraph("Clinical Insights:", styles['Heading3']))
        action_elements.append(Paragraph(action['insights'].replace('\n', '<br/>'), normal_style))
        action_elements.append(Spacer(1, 0.1 * inch))
        
        if 'clip_path' in action and action['clip_path']:
            action_elements.append(Paragraph(f"Action Clip Reference: <a href='file://{action['clip_path']}'>{action['clip_path']}</a>", normal_style))
        
        action_elements.append(Spacer(1, 0.4 * inch))
        story.append(KeepTogether(action_elements))
        
    doc.build(story)
