import scrapy

class DmozSpider(scrapy.Spider):
    name = "zhihu"
    allowed_domains = ["zhihu.com"]
    start_urls = [
        "https://www.zhihu.com/explore"
    ]

    def parse(self, response):
        filename = response.url.split("/")[-1] + '.html'
        filename = "zhihu/question/"+filename
        with open(filename, 'wb') as f:
            f.write(response.body)
        for href in response.css("a.question_link::attr('href')"):
            print "href:::::"+href.extract()
            url = "https://www.zhihu.com"+href.extract()
            yield scrapy.Request(url, callback=self.parse)



