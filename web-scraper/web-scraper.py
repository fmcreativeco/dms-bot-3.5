import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class VendorURLSpider(CrawlSpider):
    name = "vendor_url_spider"
    
    # Restrict the domains the spider is allowed to crawl
    allowed_domains = ["dms.myflorida.com"]
    
    # Start crawling from these URLs
    start_urls = [
        "https://www.dms.myflorida.com/business_operations/state_purchasing/state_contracts_and_agreements/alternate_contract_source/cloud_solutions/contractors",
        "https://www.dms.myflorida.com/business_operations/state_purchasing/state_contracts_and_agreements/alternate_contract_source/software_value_added_reseller_svar/contractors"
    ]
    
    # Rules for the spider to crawl links within pages
    rules = [
        # Spider can only crawl links that follow the defined patterns in "allow" parameter and then
        # call the the "parse_item" method for each link followed
        Rule(LinkExtractor(allow=("/business_operations/state_purchasing/state_contracts_and_agreements/alternate_contract_source/cloud_solutions/contractors", "/business_operations/state_purchasing/state_contracts_and_agreements/alternate_contract_source/software_value_added_reseller_svar/contractors")), callback='parse_item', follow=True)
    ]
    
    def parse_item(self, response):
        # Extract the contract title and number 
        contract_title = response.css('h1::text').get()
        contract_number = response.css('h1 span.contractNumber::text').get()
        
        # Check if on a contract source page
        if contract_title and contract_number:
            # Extract all the URLs from the response page
            contractors_urls = response.css('a::attr(href)').getall()
            # Filter out URLs that don't follow the "/contractors/contractors_" pattern
            contractors_urls = [url for url in contractors_urls if "/contractors/contractors_" in url]
            
            # Return an item with the contract title, number and contractor URLs
            yield {
                'title': contract_title,
                'number': contract_number,
                'contractors': [{'url': response.urljoin(url)} for url in contractors_urls]
            }
                
# Create a CrawlerProcess with defined settings
process = CrawlerProcess(settings={
    # Scraped items will be saved in the "vendor_urls.json" file in JSON format
    "FEEDS": {
        "vendor_urls.json": {"format": "json"},
    },
    # Set log level to INFO so only relevant information will be printed
    'LOG_LEVEL': 'INFO',
    # DUPEFILTER_CLASS is set to RFPDupeFilter, which filters duplicate requests
    'DUPEFILTER_CLASS': 'scrapy.dupefilters.RFPDupeFilter',
})

# Start the crawling process
print("Starting the crawl...")
process.crawl(VendorURLSpider)
process.start() 
print("Crawl finished. Results are saved in vendor_urls.json")
