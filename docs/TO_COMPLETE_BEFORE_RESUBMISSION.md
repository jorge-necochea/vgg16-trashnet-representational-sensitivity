# Do not forget before journal resubmission

1. Update the Methods and response to R6 with the recovered JPG settings:
   Pillow `Image.save()`, implicit JPEG quality 75, and no explicitly supplied
   subsampling, optimization, or progressive option. State that the exact
   historical Pillow/libjpeg version was not logged.
2. Push the final repository to GitHub.
3. Create a Zenodo release and replace `[DOI to be added]` in the Data
   Availability Statement.
4. Add the final repository URL and Zenodo DOI to the manuscript and response
   letter.
5. Add the detailed 822 JPG–PNG derived comparison table if the original CSV
   is recovered. The current assembled files contain only the aggregate
   storage-format summary reported in the manuscript, not the 60-cell source
   table.
