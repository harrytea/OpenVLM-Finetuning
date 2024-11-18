/usr/local/qcloud/stargate/admin/uninstall.sh
/usr/local/qcloud/YunJing/uninst.sh
/usr/local/qcloud/monitor/barad/admin/uninstall.sh

systemctl stop tat_agent
systemctl disable tat_agent

rm -rf /etc/systemd/system/tat_agent.service
rm -rf /usr/local/qcloud

sudo sed -i '/qcloud/d' /etc/rc.local

