#!/bin/bash

# Define variables
docker stop shopping_admin
docker remove shopping_admin
docker run --name shopping_admin -p 7780:80 -d shopping_admin_final_0719
# Wait for the container to start
echo "Waiting for the container to be ready..."
sleep 10  # Adjust this time if necessary

docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_is_forced 0
docker exec shopping_admin php /var/www/magento2/bin/magento config:set admin/security/password_lifetime 0
docker exec shopping_admin /var/www/magento2/bin/magento setup:store-config:set --base-url="http://localhost:7780"
docker exec shopping_admin mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="http://localhost:7780/" WHERE path = "web/secure/base_url";'
docker exec shopping_admin /var/www/magento2/bin/magento cache:flush
# wait ~15 secs for all services to start
# sleep 60
