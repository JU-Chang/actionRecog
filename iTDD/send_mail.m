
mail = 'zhang_yinzhu@sina.com';  % ①邮箱地址
password = '133781'; % ②密码a
% 服务器设置
setpref('Internet','E_mail',mail);
setpref('Internet','SMTP_Server','smtp.sina.com'); % ③SMTP服务器
setpref('Internet','SMTP_Username',mail);
setpref('Internet','SMTP_Password',password);
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');



% 收件人
receiver='zhang-yinzhu@qq.com'; 
% 邮件标题
mailtitle='!!!** ACTION RECOGNITION';
try
    extract_fv_i(3,'spatial')

    % 邮件内容
    mailcontent='mission compeleted!';
%     mailcontent=['mission compeleted!  ',...
%         'accuracy=',num2str(accuracy)];
    % 发送
    sendmail(receiver, mailtitle, mailcontent); 
catch
    mailcontent=['error'];
    % 发送
    sendmail(receiver, mailtitle, mailcontent); 
end
