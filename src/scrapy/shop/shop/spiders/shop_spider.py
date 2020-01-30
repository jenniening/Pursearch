# -*- coding: utf-8 -*-
import scrapy


class ShopSpiderSpider(scrapy.Spider):
    name = 'shop_spider'
    allowed_domains = ['us.shein.com']
    #allowed_domains = ['https://www.saksfifthavenue.com']
    start_urls = ["https://us.shein.com/Women-Bags-c-2043.html?icn=women-bags&ici=us_tab01navbar07menu05&scici=navbar_2~~tab01navbar07menu05~~7_5~~real_2043~~SPcCccWomenCategory_default~~0~~0&sort=8&child_id=2155&page=3"]
    #start_urls = ["https://www.saksfifthavenue.com/Handbags/Top-Handles-and-Satchels/shop/_/N-52k0hi/Ne-6lvnb5?FOLDER%3C%3Efolder_id=2534374306623862&Nao=900&Ns=P_gwp_flag%7C0%7C%7CP_bestsellers_units%7C1%7C%7CP_brandname%7C%7CP_arrivaldate%7C1%7C%7CP_product_code%7C1"]
    def parse(self, response):
        print("procesing:"+response.url)
        html = ["https://us.shein.com" + i for i in response.css("div.j-switch-color-wrap").xpath("a/@href").extract()]
        #html = [i for i in response.css("div.pa-quickview-redesign--visible").css("div.pa-product-large").css("div.image-container-large").xpath("a/@href").extract() if i != "#"]       
        #price = response.css("div.pa-quickview-redesign--visible").css("div.pa-product-large").css("div.product-text").css("span.product-price::text").extract()
        name = response.css("div.j-switch-color-wrap").css("a.c-goodsitem__ratioimg.j-item-msg.j-item-msg-a.j-expose__target-goods-img::attr(title)").extract()
        #name = response.css("div.pa-quickview-redesign--visible").css("div.pa-product-large").css("div.product-text").css("p.product-description::text").extract()
        
        #brand = response.css("div.pa-quickview-redesign--visible").css("div.pa-product-large").css("div.product-text").css("span.product-designer-name::text").extract()
        image = [i[2:] for i in response.css("div.j-switch-color-wrap").css("img.j-verlok-lazy::attr(data-src)").extract()]
        #image = response.css("div.pa-quickview-redesign--visible").css("div.pa-product-large").css("div.image-container-large").css("img::attr(src)").extract()

        row_data = zip(html, name, image)

        for item in row_data:
            scraped_info = {
                #key:value
                'page':item[0],
                'product' : item[1], #item[0] means product in the list and so on, index tells what value to assign
                'brand' : "shein",
                'price' : "less $50",
                'image' : item[2],
                'type' : 'shein_satchel'
            }
            yield scraped_info


         
