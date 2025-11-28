# src/data_generator.py
"""
Synthetic Data Generator for Document Classification
Creates realistic-looking document images with variations
"""

import os
import numpy as np
import cv2
import random
from typing import List, Dict, Tuple
from datetime import datetime, timedelta  # FIXED: Import timedelta

class DocumentDataGenerator:
    """
    Generates synthetic document images with variations
    """
    
    def __init__(self):
        self.classes = ['invoice', 'receipt', 'contract', 'form', 'letter']
        self.colors = {
            'invoice': (200, 200, 255),    # Light blue
            'receipt': (200, 255, 200),    # Light green  
            'contract': (255, 255, 200),   # Light yellow
            'form': (255, 200, 200),       # Light red
            'letter': (255, 255, 255)      # White
        }
        
        # Different templates for each class to create variety
        self.invoice_templates = self._create_invoice_templates()
        self.receipt_templates = self._create_receipt_templates()
        self.contract_templates = self._create_contract_templates()
        self.form_templates = self._create_form_templates()
        self.letter_templates = self._create_letter_templates()
    
    def _create_invoice_templates(self):
        """Create different invoice templates"""
        return [
            {
                'company': 'Tech Solutions Inc.',
                'items': [
                    ('Software License', 2, 299.99),
                    ('Technical Support', 1, 150.00),
                    ('Hardware Upgrade', 1, 450.50)
                ]
            },
            {
                'company': 'Global Services Ltd.',
                'items': [
                    ('Consulting Hours', 10, 125.00),
                    ('Project Management', 1, 500.00),
                    ('Travel Expenses', 1, 275.75)
                ]
            },
            {
                'company': 'Creative Designs Co.',
                'items': [
                    ('Website Design', 1, 1200.00),
                    ('Logo Creation', 2, 350.00),
                    ('Marketing Materials', 1, 450.25)
                ]
            },
            {
                'company': 'Office Supplies Corp.',
                'items': [
                    ('Paper Ream', 5, 25.99),
                    ('Pens (Box)', 2, 15.50),
                    ('Printers', 1, 289.99)
                ]
            }
        ]
    
    def _create_receipt_templates(self):
        """Create different receipt templates"""
        return [
            {
                'store': 'SuperMart Grocery',
                'items': [
                    ('Milk', 2.99),
                    ('Bread', 3.49),
                    ('Eggs', 4.99),
                    ('Cheese', 5.25)
                ]
            },
            {
                'store': 'TechGadget Store',
                'items': [
                    ('USB Cable', 12.99),
                    ('Mouse', 25.50),
                    ('Keyboard', 45.75),
                    ('Headphones', 89.99)
                ]
            },
            {
                'store': 'Fashion Boutique',
                'items': [
                    ('T-Shirt', 24.99),
                    ('Jeans', 49.99),
                    ('Shoes', 79.50),
                    ('Accessories', 15.25)
                ]
            },
            {
                'store': 'Home & Garden',
                'items': [
                    ('Plants', 12.99),
                    ('Tools', 18.50),
                    ('Decor', 32.75),
                    ('Furniture', 199.99)
                ]
            }
        ]
    
    def _create_contract_templates(self):
        """Create different contract templates"""
        return [
            {
                'title': 'EMPLOYMENT AGREEMENT',
                'parties': ['Tech Innovations LLC', 'John Smith'],
                'terms': ['12 months employment', 'Health benefits', 'Vacation days']
            },
            {
                'title': 'CONSULTING SERVICES',
                'parties': ['Business Solutions Inc.', 'Consulting Partners'],
                'terms': ['Project based work', 'Hourly billing', 'Monthly reports']
            },
            {
                'title': 'LEASE AGREEMENT',
                'parties': ['Property Management Co.', 'Tenant Name'],
                'terms': ['12 month lease', 'Security deposit', 'Maintenance terms']
            },
            {
                'title': 'NON-DISCLOSURE AGREEMENT',
                'parties': ['Company XYZ', 'Contractor Name'],
                'terms': ['Confidentiality terms', 'Duration 2 years', 'Legal jurisdiction']
            }
        ]
    
    def _create_form_templates(self):
        """Create different form templates"""
        return [
            {
                'title': 'JOB APPLICATION',
                'fields': ['Full Name', 'Position', 'Experience', 'Education']
            },
            {
                'title': 'MEMBERSHIP FORM',
                'fields': ['Member Name', 'Address', 'Phone', 'Email', 'Membership Type']
            },
            {
                'title': 'SURVEY FORM',
                'fields': ['Age Group', 'Occupation', 'Income Range', 'Preferences']
            },
            {
                'title': 'REGISTRATION FORM',
                'fields': ['First Name', 'Last Name', 'Date of Birth', 'Emergency Contact']
            }
        ]
    
    def _create_letter_templates(self):
        """Create different letter templates"""
        return [
            {
                'type': 'BUSINESS PROPOSAL',
                'recipient': 'Investment Partners',
                'content': 'We are excited to present this business opportunity...'
            },
            {
                'type': 'COVER LETTER',
                'recipient': 'Hiring Manager',
                'content': 'I am writing to express my interest in the position...'
            },
            {
                'type': 'THANK YOU LETTER',
                'recipient': 'Valued Customer',
                'content': 'Thank you for your recent business with our company...'
            },
            {
                'type': 'INVITATION LETTER',
                'recipient': 'Event Attendees',
                'content': 'You are cordially invited to attend our annual conference...'
            }
        ]
    
    def generate_complete_dataset(self, 
                                base_dir: str = 'data',
                                train_samples: int = 400,
                                test_samples: int = 100,
                                val_samples: int = 100,
                                img_size: Tuple[int, int] = (224, 224)) -> Dict[str, int]:
        """
        Generate complete dataset with variations
        """
        samples_per_class = {
            'train': train_samples // len(self.classes),
            'test': test_samples // len(self.classes),
            'val': val_samples // len(self.classes)
        }
        
        print(f"📊 Generating varied dataset:")
        print(f"   Training: {samples_per_class['train']} unique samples per class")
        print(f"   Testing: {samples_per_class['test']} unique samples per class")
        print(f"   Validation: {samples_per_class['val']} unique samples per class")
        
        generated_counts = {}
        
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(base_dir, split)
            count = self._generate_varied_split(
                split_dir=split_dir,
                samples_per_class=samples_per_class[split],
                img_size=img_size,
                split_name=split
            )
            generated_counts[split] = count
            
        return generated_counts
    
    def _generate_varied_split(self, 
                             split_dir: str,
                             samples_per_class: int,
                             img_size: Tuple[int, int],
                             split_name: str) -> int:
        """
        Generate varied images for a specific split
        """
        print(f"\n📁 Generating varied {split_name} data...")
        
        total_generated = 0
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            class_generated = 0
            for i in range(samples_per_class):
                try:
                    # Generate unique document image with variations
                    img = self._generate_varied_document(class_name, img_size, i)
                    
                    # Add split-specific visual variations
                    img = self._add_split_variations(img, split_name, i)
                    
                    # Save image
                    filename = f"{class_name}_{i:04d}.jpg"
                    filepath = os.path.join(class_dir, filename)
                    cv2.imwrite(filepath, img)
                    
                    class_generated += 1
                    total_generated += 1
                    
                except Exception as e:
                    print(f"⚠️  Failed to generate {class_name} image {i}: {e}")
                    continue
            
            print(f"   ✅ {class_name}: {class_generated} unique images")
            
        print(f"   📈 Total {split_name}: {total_generated} varied images")
        return total_generated
    
    def _generate_varied_document(self, 
                                doc_type: str, 
                                img_size: Tuple[int, int],
                                index: int) -> np.ndarray:
        """
        Generate a unique document with variations
        """
        # Base image with slight color variations
        base_color = list(self.colors[doc_type])
        # Add slight color variation
        color_variation = random.randint(-10, 10)
        base_color = [max(0, min(255, c + color_variation)) for c in base_color]
        
        img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8)
        img[:, :] = base_color
        
        # Choose random template for this document
        if doc_type == 'invoice':
            template = random.choice(self.invoice_templates)
            return self._create_varied_invoice(img, index, template)
        elif doc_type == 'receipt':
            template = random.choice(self.receipt_templates)
            return self._create_varied_receipt(img, index, template)
        elif doc_type == 'contract':
            template = random.choice(self.contract_templates)
            return self._create_varied_contract(img, index, template)
        elif doc_type == 'form':
            template = random.choice(self.form_templates)
            return self._create_varied_form(img, index, template)
        elif doc_type == 'letter':
            template = random.choice(self.letter_templates)
            return self._create_varied_letter(img, index, template)
        else:
            return img
    
    def _create_varied_invoice(self, img: np.ndarray, index: int, template: dict) -> np.ndarray:
        """Create varied invoice document"""
        h, w = img.shape[:2]
        
        # Random invoice number and date
        invoice_no = 10000 + index + random.randint(1000, 9999)
        days_ago = random.randint(1, 30)
        invoice_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")  # FIXED
        
        # Header with template company
        cv2.rectangle(img, (20, 20), (w-20, 70), (230, 230, 230), -1)
        cv2.putText(img, f'INVOICE #{invoice_no}', (w//2 - 80, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, f'Date: {invoice_date}', (50, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Company info from template
        cv2.putText(img, f'From: {template["company"]}', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, 'To: Various Clients', (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Table header
        cv2.rectangle(img, (50, 150), (w-50, 180), (220, 220, 220), -1)
        cv2.putText(img, 'Description', (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, 'Qty', (w-150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, 'Amount', (w-80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Table rows from template with random variations
        for i, (item, qty, price) in enumerate(template['items']):
            y_pos = 210 + i * 30
            cv2.line(img, (50, y_pos), (w-50, y_pos), (180, 180, 180), 1)
            cv2.putText(img, item, (60, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.putText(img, str(qty), (w-150, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            # Add small random price variation
            varied_price = price * random.uniform(0.9, 1.1)
            cv2.putText(img, f'${varied_price:.2f}', (w-80, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Calculate total
        total = sum(item[2] for item in template['items'])
        cv2.line(img, (w-200, h-80), (w-50, h-80), (0, 0, 0), 2)
        cv2.putText(img, f'TOTAL: ${total:.2f}', (w-180, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return img
    
    def _create_varied_receipt(self, img: np.ndarray, index: int, template: dict) -> np.ndarray:
        """Create varied receipt document"""
        h, w = img.shape[:2]
        
        # Store header from template
        cv2.putText(img, template['store'], (w//2 - 60, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, 'Various Locations', (w//2 - 70, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        receipt_no = 20000 + index + random.randint(100, 999)
        cv2.putText(img, f'Receipt #{receipt_no}', (w//2 - 70, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.line(img, (20, 80), (w-20, 80), (0, 0, 0), 1)
        
        # Items from template with random quantities
        y_start = 110
        for i, (item, base_price) in enumerate(template['items']):
            y_pos = y_start + i * 25
            # Random quantity
            qty = random.randint(1, 3)
            price = base_price * qty * random.uniform(0.95, 1.05)  # Small price variation
            
            cv2.putText(img, f'{item} x{qty}', (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.putText(img, f'${price:.2f}', (w-80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        # Calculations with variations
        items_with_prices = [(item[0], item[1] * random.randint(1, 3) * random.uniform(0.95, 1.05)) 
                           for item in template['items']]
        subtotal = sum(price for _, price in items_with_prices)
        tax_rate = random.uniform(0.07, 0.09)  # Varying tax rates
        tax = subtotal * tax_rate
        total = subtotal + tax
        
        y_calc = y_start + len(template['items']) * 25 + 20
        cv2.line(img, (20, y_calc), (w-20, y_calc), (100, 100, 100), 1)
        
        cv2.putText(img, f'Subtotal: ${subtotal:.2f}', (30, y_calc + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, f'Tax: ${tax:.2f}', (30, y_calc + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        cv2.line(img, (20, y_calc + 65), (w-20, y_calc + 65), (0, 0, 0), 2)
        cv2.putText(img, f'TOTAL: ${total:.2f}', (30, y_calc + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Random footer messages
        footers = [
            'Thank you for your business!',
            'We appreciate your visit!',
            'Come again soon!',
            'Have a great day!'
        ]
        cv2.putText(img, random.choice(footers), (w//2 - 100, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return img
    
    def _create_varied_contract(self, img: np.ndarray, index: int, template: dict) -> np.ndarray:
        """Create varied contract document"""
        h, w = img.shape[:2]
        
        # Title from template
        cv2.putText(img, template['title'], (w//2 - 120, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.line(img, (50, 50), (w-50, 50), (0, 0, 0), 1)
        
        # Contract number and date
        contract_no = 30000 + index + random.randint(100, 999)
        days_ago = random.randint(1, 365)
        contract_date = (datetime.now() - timedelta(days=days_ago)).strftime("%B %d, %Y")  # FIXED
        
        cv2.putText(img, f'Contract No: CT-{contract_no}', (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, f'Effective Date: {contract_date}', 
                   (w-250, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Contract content with template parties
        parties_text = [
            "THIS AGREEMENT is made and entered into as of the date first above",
            "written, by and between:",
            "",
            f"PARTY A: {template['parties'][0]}",
            f"PARTY B: {template['parties'][1]}",
            "",
            "ARTICLE 1 - TERMS AND CONDITIONS",
        ]
        
        # Add template terms
        terms_text = template['terms'][:3]  # Take first 3 terms
        
        y_pos = 120
        for line in parties_text + terms_text:
            if line:
                cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            y_pos += 20
        
        # Add some random legal text
        legal_text = [
            "",
            "All parties agree to abide by the terms outlined herein.",
            "This agreement constitutes the entire understanding between parties.",
            "Any modifications must be made in writing and signed by both parties."
        ]
        
        for line in legal_text:
            if line:
                cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
            y_pos += 18
        
        # Signature lines with random names
        cv2.putText(img, "IN WITNESS WHEREOF, the parties have executed this Agreement.", 
                   (50, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
        
        cv2.line(img, (50, h-70), (200, h-70), (0, 0, 0), 1)
        cv2.putText(img, template['parties'][0], (50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        cv2.line(img, (w-200, h-70), (w-50, h-70), (0, 0, 0), 1)
        cv2.putText(img, template['parties'][1], (w-200, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return img
    
    def _create_varied_form(self, img: np.ndarray, index: int, template: dict) -> np.ndarray:
        """Create varied form document"""
        h, w = img.shape[:2]
        
        # Title from template
        cv2.putText(img, template['title'], (w//2 - 80, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        form_id = 40000 + index + random.randint(10, 99)
        cv2.putText(img, f'Form ID: FM-{form_id}', (w-150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Form fields from template
        y_start = 80
        for i, field in enumerate(template['fields']):
            y_pos = y_start + i * 40
            cv2.putText(img, field + ':', (30, y_pos-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            # Vary field width
            field_width = random.randint(200, 280)
            cv2.rectangle(img, (30, y_pos), (30 + field_width, y_pos + 25), (0, 0, 0), 1)
        
        # Random additional elements
        y_extra = y_start + len(template['fields']) * 40 + 20
        
        # Random choice of additional elements
        extra_elements = random.choice([
            ['radio_buttons', 'checkboxes'],
            ['dropdown', 'signature'],
            ['date_picker', 'file_upload']
        ])
        
        if 'radio_buttons' in extra_elements:
            cv2.putText(img, 'Options:', (30, y_extra), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.circle(img, (100, y_extra-5), 6, (0, 0, 0), 1)
            cv2.putText(img, 'Option A', (115, y_extra), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            cv2.circle(img, (200, y_extra-5), 6, (0, 0, 0), 1)
            cv2.putText(img, 'Option B', (215, y_extra), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            y_extra += 30
        
        if 'checkboxes' in extra_elements:
            cv2.rectangle(img, (30, y_extra), (45, y_extra+15), (0, 0, 0), 1)
            cv2.putText(img, 'I agree to terms', (50, y_extra+15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            y_extra += 25
        
        # Submit button with random text
        button_texts = ['SUBMIT', 'SEND', 'APPLY', 'REGISTER', 'CONTINUE']
        cv2.rectangle(img, (w//2 - 50, h-50), (w//2 + 50, h-20), (200, 200, 200), -1)
        cv2.rectangle(img, (w//2 - 50, h-50), (w//2 + 50, h-20), (0, 0, 0), 2)
        cv2.putText(img, random.choice(button_texts), (w//2 - 35, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img
    
    def _create_varied_letter(self, img: np.ndarray, index: int, template: dict) -> np.ndarray:
        """Create varied letter document"""
        h, w = img.shape[:2]
        
        # Letterhead with variations
        company_names = ['Business Corp.', 'Global Enterprises', 'Innovation Labs', 'Solutions Inc.']
        addresses = [
            '123 Corporate Plaza, Suite 500',
            '456 Business Avenue, Floor 3', 
            '789 Innovation Street',
            '321 Commerce Center'
        ]
        
        cv2.putText(img, random.choice(company_names), (w//2 - 70, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, random.choice(addresses), (w//2 - 120, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(img, 'City, State 12345 • (555) 123-4567', (w//2 - 120, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        # Date and recipient
        days_ago = random.randint(1, 60)
        letter_date = (datetime.now() - timedelta(days=days_ago)).strftime("%B %d, %Y")  # FIXED
        cv2.putText(img, letter_date, (w-150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        
        recipients = ['Client Services', 'Hiring Manager', 'Investment Committee', 'Board of Directors']
        cv2.putText(img, f'To: {random.choice(recipients)}', (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, template['recipient'], (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, 'Various Locations', (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Salutation
        salutations = ['Dear Sir/Madam,', 'To Whom It May Concern,', 'Greetings,', 'Hello,']
        cv2.putText(img, random.choice(salutations), (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Letter body with template content
        paragraphs = [
            template['content'],
            "",
            "We believe this opportunity aligns with our mutual interests",
            "and would appreciate the chance to discuss this further.",
            "",
            "Please feel free to contact us at your earliest convenience.",
        ]
        
        y_pos = 220
        for para in paragraphs:
            if para:
                # Vary line lengths for more natural look
                words = para.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + " " + word) <= 40:
                        current_line += " " + word if current_line else word
                    else:
                        lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                for line in lines:
                    cv2.putText(img, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                    y_pos += 18
            else:
                y_pos += 15
        
        # Closing with random names
        closings = ['Sincerely,', 'Best regards,', 'Respectfully,', 'Yours truly,']
        names = ['John A. Smith', 'Sarah Johnson', 'Michael Brown', 'Emily Davis']
        titles = [
            'Director of Business Development',
            'Senior Manager', 
            'Project Lead',
            'Account Executive'
        ]
        
        cv2.putText(img, random.choice(closings), (50, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(img, random.choice(names), (50, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, random.choice(titles), (50, h-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return img

    # Keep the existing variation methods
    def _add_split_variations(self, img: np.ndarray, split_name: str, index: int) -> np.ndarray:
        """Add visual variations"""
        variations = [
            self._add_noise,
            self._adjust_brightness,
            self._add_blur,
            self._adjust_contrast,
            self._add_light_noise,
            self._add_light_blur
        ]
        
        # Apply random variations
        num_variations = random.randint(1, 3)
        selected_variations = random.sample(variations, num_variations)
        
        for variation_func in selected_variations:
            img = variation_func(img, index)
            
        return img
    
    def _add_noise(self, img: np.ndarray, index: int) -> np.ndarray:
        noise = np.random.normal(0, random.randint(5, 20), img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    def _add_light_noise(self, img: np.ndarray, index: int) -> np.ndarray:
        noise = np.random.normal(0, random.randint(3, 10), img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    def _adjust_brightness(self, img: np.ndarray, index: int) -> np.ndarray:
        brightness = random.randint(-40, 40)
        return cv2.add(img, brightness)
    
    def _add_blur(self, img: np.ndarray, index: int) -> np.ndarray:
        # Compute a kernel size that scales with image size. Kernel must be
        # odd and >= 3. Use a small kernel for small images and larger for
        # high-resolution images.
        h, w = img.shape[:2]
        base = max(3, min(h, w) // 512)  # scale factor
        kernel = base if base % 2 == 1 else base + 1
        if kernel < 3:
            kernel = 3
        return cv2.GaussianBlur(img, (kernel, kernel), 0)
    
    def _add_light_blur(self, img: np.ndarray, index: int) -> np.ndarray:
        # Lighter blur: smaller kernel but still odd and >=3. Scale with image.
        h, w = img.shape[:2]
        kernel = max(3, (min(h, w) // 1024) * 2 + 1)
        if kernel < 3:
            kernel = 3
        return cv2.GaussianBlur(img, (kernel, kernel), 0)
    
    def _adjust_contrast(self, img: np.ndarray, index: int) -> np.ndarray:
        alpha = random.uniform(0.7, 1.3)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


# Utility functions
def generate_complete_dataset(base_dir: str = 'data', 
                             train_samples: int = 400,
                             test_samples: int = 100,
                             val_samples: int = 100,
                             img_size: Tuple[int, int] = (224, 224)) -> Dict[str, int]:
    """Module-level helper that forwards `img_size` to the class method.

    `img_size` is (height, width). To request full-size A4 at 300 DPI,
    call with `img_size=(3508, 2480)` or use the CLI `--full-size` flag.
    """
    generator = DocumentDataGenerator()
    return generator.generate_complete_dataset(base_dir, train_samples, test_samples, val_samples, img_size)

def generate_training_data_only(data_dir: str = 'data/raw/documents',
                              samples_per_class: int = 100) -> int:
    generator = DocumentDataGenerator()
    return generator.generate_training_data_only(data_dir, samples_per_class)