#!/usr/bin/env python3
"""One-page forecaster reference for the learned LPI and dry-lightning screen."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
OUT=Path('output/pdf/lpi_v4_dry_lightning_quick_reference.pdf')
NAVY=HexColor('#17324D');TEAL=HexColor('#007F82');INK=HexColor('#243441');MUTED=HexColor('#52616B')
BODY=ParagraphStyle('body',fontName='Helvetica',fontSize=9.3,leading=12,textColor=INK)
SMALL=ParagraphStyle('small',fontName='Helvetica',fontSize=8.5,leading=10.5,textColor=MUTED)
TITLE=ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=NAVY)

def para(c,text,x,top,w,style=BODY,bottom=None):
 p=Paragraph(text,style);_,h=p.wrap(w,1000)
 if bottom is not None:assert top-h>=bottom,(text[:60],top-h,bottom)
 p.drawOn(c,x,top-h);return h

def box(c,x,y,w,h,color):
 c.setFillColor(HexColor(color));c.setStrokeColor(HexColor('#C9D3D9'));c.setLineWidth(.6);c.roundRect(x,y,w,h,7,fill=1,stroke=1)

def build():
 OUT.parent.mkdir(parents=True,exist_ok=True);c=canvas.Canvas(str(OUT),pagesize=letter);c.setTitle('LPI and Dry Lightning - Quick Reference');c.setAuthor('BC Forecast Graphics');c.setSubject('Learned LPI probability and dry-lightning interpretation')
 c.setFillColor(NAVY);c.rect(0,722,612,70,fill=1,stroke=0);c.setFillColor(white);c.setFont('Helvetica-Bold',22);c.drawString(30,757,'LPI and Dry Lightning')
 c.setFont('Helvetica',10);c.drawString(30,737,'Quick reference | BC continental HRDPS | 7 October 2026')
 box(c,30,631,552,79,'#EDF7F6');para(c,'WHAT THE LPI VALUE MEANS',43,699,526,TITLE)
 para(c,'LPI is a <b>0-100% model estimate of lightning occurring anywhere within 30 km</b> during the stated forecast period. LPI 40 means an estimated 40% chance of at least one lightning occurrence nearby. It does not describe flash density or the exact strike location. <b>24-hour and 3-hour probabilities are calculated separately.</b>',43,678,526,bottom=642)
 box(c,30,525,552,94,'#F6F8F9');para(c,'HOW THE MODEL COMBINES THE INGREDIENTS',43,607,526,TITLE)
 para(c,'Hourly ingredients become <b>20 summaries</b>: peaks, means, fractions of favourable hours, and combinations that occur together. Each summary is smoothed with a <b>20 km Gaussian sigma</b> on a 5 km analysis grid. The model combines these with <b>32 nonlinear ingredient patterns</b> and converts the learned weighted result to a probability.',43,585,526,bottom=537)
 # Two readable ingredient columns.
 box(c,30,309,552,204,'#EDF7F6');para(c,'WHAT GOES INTO THE LPI',43,501,526,TITLE)
 left='<b>Instability</b><br/>Most-unstable lifted index (MU-LI): scaled from +1 to -5 C. Most-unstable CAPE: 75-800 J/kg.<br/><br/><b>Charging environment</b><br/>Humidity in the 0 to -20 C layer: 45-80%, weighted most strongly near -15 C. Its weighted pressure depth: 35-150 hPa. Mid-level humidity (-5 to -30 C): 35-75%.<br/><br/>These describe the potential for an unstable, humid mixed-phase cloud.'
 right='<b>Storm development and rain</b><br/>Resolved ascent at 500/700 hPa: 0.005-0.050 m/s. Trailing 3-hour rain: 0.05-1.5 mm. Rain rate: 0.02-0.8 mm/h.<br/><br/><b>Humidity and persistence</b><br/>Surface and subcloud humidity; average instability, charging and ascent; fractions of hours with instability/ascent; same-hour storm ingredient combinations.<br/><br/>Ranges scale inputs to 0-1. Learned weights and patterns set their combined influence.'
 para(c,left,43,478,248,bottom=320);para(c,right,310,478,259,bottom=320)
 box(c,30,135,552,162,'#FFF4E6');para(c,'DRY-LIGHTNING SCREENING',43,285,526,TITLE)
 dry='<b>Hourly dryness:</b> RHdry = min(surface RH, subcloud RH). The dryness factor starts below <b>55% RH</b> and reaches full strength at <b>30% RH</b>. Subcloud RH averages surface RH and valid above-ground RH at 850/800/750/700 hPa.<br/><br/><b>Three-hour dry score:</b> where 3-hour LPI is at least <b>20%</b>, multiply LPI by the strongest hourly dryness factor in that block. Reduce the score as trailing 3-hour rain rises from <b>0.25 to 2.5 mm</b>; the rain factor reaches zero at 2.5 mm.<br/><br/><b>Asterisks appear at a dry score of 15 or more.</b> In the BC two-panel display they are grey below LPI 60% and black at 60% or more. This score is a <b>heuristic screen</b>, not a dry-lightning or fire-ignition probability. Check local rainfall, fuels and observations.'
 para(c,dry,43,263,526,bottom=146)
 para(c,'READING THE DISPLAY AND USING THE GUIDANCE',30,120,552,TITLE)
 para(c,'The right panel has fire-danger shading and purple-to-magenta LPI contours at <b>20, 40, 60 and 80%</b>. Cyan/teal dots show 3-hour rain at <b>2.5/10 mm</b>. The left panel shows humidity and gusts. The daily product covers <b>12Z to the next 12Z</b>; a three-hour map covers the block ending at its valid time. Confirm the period before interpreting a value.',30,100,552,SMALL,bottom=56)
 para(c,'Deployment scope: 12Z continental HRDPS, first 24 forecast hours. Experimental partial-season calibration. Missing ingredients remain missing; blank areas are not a forecast of zero lightning.',30,48,552,SMALL,bottom=24)
 c.setFillColor(NAVY);c.rect(0,0,612,17,fill=1,stroke=0);c.setFillColor(white);c.setFont('Helvetica',7);c.drawString(30,5,'BC LPI | bc_lpi_v4_random32 | ECCC HRDPS inputs');c.drawRightString(582,5,'Forecaster quick reference | 1 / 1');c.showPage();c.save();print(OUT)
if __name__=='__main__':build()
