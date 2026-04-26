import re
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER


def _parse_insights(text, styles):
    elements = []
    heading_pattern = re.compile(r'^\*\*(.+?)\*\*$', re.MULTILINE)
    parts = heading_pattern.split(text)
    
    is_heading = False
    for part in parts:
        part = part.strip()
        if not part:
            is_heading = not is_heading
            continue
        if is_heading:
            elements.append(Spacer(1, 0.08 * inch))
            elements.append(Paragraph(part, styles['Heading3']))
        else:
            cleaned = part.replace('\n', '<br/>')
            elements.append(Paragraph(cleaned, styles['Normal']))
        is_heading = not is_heading
    return elements


def generate_surgical_report(output_path, global_summary, actions_data):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    ts_style = ParagraphStyle('TSStyle', parent=normal_style, alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=8)
    
    story = []
    
    story.append(Paragraph("Surgical Action & Segmentation Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    story.append(Paragraph("Detected Surgical Actions", heading_style))
    if actions_data:
        for idx, action in enumerate(actions_data, 1):
            ts = action.get('timestamp', 'N/A')
            name = action.get('action', 'Unknown')
            story.append(Paragraph(f"{idx}. [{ts}] {name}", normal_style))
    else:
        story.append(Paragraph(global_summary, normal_style))
    story.append(Spacer(1, 0.5 * inch))
    
    for action in actions_data:
        action_elements = []
        
        ts = action.get('timestamp', 'N/A')
        name = action.get('action', 'Unknown')
        action_elements.append(Paragraph(f"[{ts}] — {name}", heading_style))
        action_elements.append(Spacer(1, 0.1 * inch))
        
        if 'keyframes_filled' in action and 'keyframe_timestamps' in action:
            kf_list = action['keyframes_filled']
            ts_list = action['keyframe_timestamps']
            if kf_list:
                img_width = 2.2 * inch
                img_height = 1.65 * inch
                
                img_row = [Image(p, width=img_width, height=img_height) for p in kf_list]
                ts_row = [Paragraph(t, ts_style) for t in ts_list]
                
                t = Table([img_row, ts_row], colWidths=[2.3 * inch] * 3)
                t.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 2)
                ]))
                action_elements.append(t)
                action_elements.append(Spacer(1, 0.15 * inch))
        elif 'keyframe_path' in action:
            action_elements.append(Image(action['keyframe_path'], width=6 * inch, height=4 * inch))
            action_elements.append(Spacer(1, 0.1 * inch))
        
        insights_text = action.get('insights', '')
        action_elements.extend(_parse_insights(insights_text, styles))
        action_elements.append(Spacer(1, 0.1 * inch))
        
        if action.get('clip_path'):
            clip_path = action['clip_path']
            action_elements.append(Paragraph(
                f"Video Clip: <a href='file://{clip_path}'>{clip_path}</a>",
                normal_style
            ))
        
        action_elements.append(Spacer(1, 0.4 * inch))
        story.append(KeepTogether(action_elements))
    
    doc.build(story)
