# Blogs Directory

This directory contains your blog files for fine-tuning dataset creation.

## File Naming Convention

Use one of these naming patterns:
- blog1.html, blog2.html, blog3.html, ...
- blog1.txt, blog2.txt, blog3.txt, ...
- blog1.htm, blog2.htm, blog3.htm, ...

## Supported Formats

- **HTML files** (.html, .htm) - Recommended
- **Text files** (.txt) - Plain text format

## How to Add Your Blogs

1. **Copy your blog content** from your website or CMS
2. **Save as HTML file** (recommended) or text file
3. **Name the file** following the convention above
4. **Place in this directory**

## Example File Structure

```
blogs/
├── blog1.html
├── blog2.html
├── blog3.html
├── ...
└── blog40.html
```

## Content Requirements

Each blog file should contain:
- Complete blog post content
- Title (in H1 tag or first line)
- Full article text
- Proper formatting

## What the Processor Does

The blog processor will:
- ✅ Extract title and keywords automatically
- ✅ Detect language (Persian/English)
- ✅ Remove images and clean HTML
- ✅ Create optimized prompts for training
- ✅ Generate perfect training dataset

## Next Steps

1. Add your 40 blog files to this directory
2. Run: `python blog_processor.py`
3. The script will create `seo_blogs_training_data.jsonl`
4. Run fine-tuning: `python fine_tuning_script.py`

## Tips for Best Results

- Use HTML format for better structure detection
- Include clear titles and headings
- Ensure content is complete and well-formatted
- Remove any sensitive or personal information
- Use your best, highest-quality blogs only
