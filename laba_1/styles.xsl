<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  
  <!-- Шаблон для всей таблицы -->
  <xsl:template match="/">
    <html>
      <head>
        <title>Таблица покупок</title>
        <style>
          table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
            padding: 8px;
          }
          th {
            background-color: #f2f2f2;
          }
          table {
            margin-top: 20px;
          }
        </style>
      </head>
      <body>
        <h2>Таблица покупок</h2>

        <!-- Вывод количества строк в таблице -->
        <p>Количество строк: <xsl:value-of select="count(/dataset/store)" /></p>

        <!-- Таблица данных -->
        <table>
          <tr>
            <th>№</th> <!-- Нумерация строк -->
            <th>Название магазина</th>
            <th>Время</th>
            <th>Широта</th>
            <th>Долгота</th>
            <th>Категория</th>
            <th>Бренд</th>
            <th>Номер карты</th>
            <th>Количество товаров</th>
            <th>Цена</th>
          </tr>
          <!-- Шаблон для каждой строки в таблице -->
          <xsl:for-each select="/dataset/store">
            <tr>
              <td><xsl:value-of select="position()" /></td> <!-- Порядковый номер строки -->
              <td><xsl:value-of select="name" /></td>
              <td><xsl:value-of select="coordinates/datetime" /></td>
              <td><xsl:value-of select="coordinates/location/latitude" /></td>
              <td><xsl:value-of select="coordinates/location/longitude" /></td>
              <td><xsl:value-of select="category" /></td>
              <td><xsl:value-of select="brand" /></td>
              <td><xsl:value-of select="card_number" /></td>
              <td><xsl:value-of select="item_count" /></td>
              <td><xsl:value-of select="price" /></td>
            </tr>
          </xsl:for-each>
        </table>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
